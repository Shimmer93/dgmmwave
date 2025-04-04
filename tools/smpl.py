"""
This file contains the definition of the SMPL model
forward: using pose and beta calculate vertex location

function get joints: calculate joints from vertex location
"""
from __future__ import division
from numpy.core.defchararray import array

import cv2
import torch
import torch.nn as nn
import numpy as np
import os, sys

root_path = os.path.dirname(__file__)

try:
    import cPickle as pickle
except ImportError:
    import pickle

#from opendr.camera import ProjectPoints
#from opendr.renderer import ColoredRenderer
#from opendr.lighting import LambertianPointLight
ProjectPoints, ColoredRenderer, LambertianPointLight = None, None, None


def save_ply(vertice, out_file):
    if type(vertice) == torch.Tensor:
        vertice = vertice.squeeze().cpu().detach().numpy()
    if vertice.ndim == 3:
        assert vertice.shape[0] == 1
        vertice = vertice.squeeze(0)
    model_file = os.path.join(root_path,'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
    with open(model_file, 'rb') as f:
        smpl_model = pickle.load(f, encoding='iso-8859-1')
        face_index = smpl_model['f'].astype(np.int64)
    face_1 = np.ones((face_index.shape[0], 1))
    face_1 *= 3
    face = np.hstack((face_1, face_index)).astype(int)
    with open(out_file, "wb") as zjy_f:
        np.savetxt(zjy_f, vertice, fmt='%f %f %f')
        np.savetxt(zjy_f, face, fmt='%d %d %d %d')
    ply_header = '''ply
format ascii 1.0
element vertex 6890
property float x
property float y
property float z
element face 13776
property list uchar int vertex_indices
end_header
    '''
    with open(out_file, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header)
        f.write(old)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2],
                         dim=1).view(B, 3, 3)
    return rotMat


def rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat2mat(quat)

SMPL_FILE = os.path.join('/home/zpengac/mmwave/dgmmwave/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
# seems all zeros.
JOINT_REGRESSOR_TRAIN_EXTRA = '/home/zpengac/mmwave/dgmmwave/J_regressor_extra.npy'

JOINTS_IDX = [8, 5, 29, 30, 4, 7, 21, 19, 17, 16, 18,
              20, 31, 32, 33, 34, 35, 36, 37, 24, 26, 25, 28, 27]

class SMPL(nn.Module):

    def __init__(self, model_file=SMPL_FILE):
        super(SMPL, self).__init__()
        with open(model_file, 'rb') as f:
            smpl_model = pickle.load(f, encoding='iso-8859-1')
        J_regressor = smpl_model['J_regressor'].tocoo()
        row = J_regressor.row
        col = J_regressor.col
        data = J_regressor.data
        i = torch.LongTensor([row, col])
        v = torch.FloatTensor(data)
        J_regressor_shape = [24, 6890]
        self.register_buffer('J_regressor', torch.sparse.FloatTensor(i, v,
                                                                     J_regressor_shape).to_dense())
        self.register_buffer(
            'weights', torch.FloatTensor(smpl_model['weights']))
        self.register_buffer(
            'posedirs', torch.FloatTensor(smpl_model['posedirs']))
        self.register_buffer(
            'v_template', torch.FloatTensor(smpl_model['v_template']))
        self.register_buffer('shapedirs',
                             torch.FloatTensor(np.array(smpl_model['shapedirs'])))
        self.register_buffer('faces',
                             torch.from_numpy(smpl_model['f'].astype(np.int64)))
        self.register_buffer('kintree_table', torch.from_numpy(
            smpl_model['kintree_table'].astype(np.int64)))
        id_to_col = {self.kintree_table[1, i].item(): i for i in
                     range(self.kintree_table.shape[1])}
        self.register_buffer('parent', torch.LongTensor(
            [id_to_col[self.kintree_table[0, it].item()] for it in
             range(1, self.kintree_table.shape[1])]))

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.translation_shape = [3]

        self.pose = torch.zeros(self.pose_shape)
        self.beta = torch.zeros(self.beta_shape)
        self.translation = torch.zeros(self.translation_shape)

        self.verts = None
        self.J = None
        self.R = None

        J_regressor_extra = torch.from_numpy(
            np.load(JOINT_REGRESSOR_TRAIN_EXTRA)).float()
        self.register_buffer('J_regressor_extra', J_regressor_extra)
        self.joints_idx = JOINTS_IDX
        self.requires_grad_(False)

    def forward(self, pose, beta):  # return vertices location
        device = pose.device
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1,
                                        10)[None, :].expand(batch_size, -1, -1)
        beta = beta[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template
        # batched sparse matmul not supported in pytorch
        J = []
        for i in range(batch_size):
            J.append(torch.matmul(self.J_regressor, v_shaped[i]))
        J = torch.stack(J, dim=0)
        # input it rotmat: (bs,24,3,3)
        if pose.ndimension() == 4:
            R = pose
        # input it rotmat: (bs,72)
        elif pose.ndimension() == 2:
            pose_cube = pose.view(-1, 3)  # (batch_size * 24, 1, 3)
            R = rodrigues(pose_cube).view(batch_size, 24, 3, 3)
            R = R.view(batch_size, 24, 3, 3)
        I_cube = torch.eye(3, dtype=R.dtype, device=R.device)
        # I_cube = torch.eye(3)[None, None, :].to(device)
        # I_cube = torch.eye(3)[None, None, :].expand(theta.shape[0], R.shape[1]-1, -1, -1)
        lrotmin = (R[:, 1:, :] - I_cube).view(batch_size, -1)
        posedirs = self.posedirs.view(-1,
                                      207)[None, :].expand(batch_size, -1, -1)
        v_posed = v_shaped + torch.matmul(posedirs, lrotmin[:, :, None]).view(-1, 6890,
                                                                              3)
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        # pad_row = torch.FloatTensor([0, 0, 0, 1]).to(device).view(1, 1, 1, 4).expand(batch_size, 24, -1, -1)
        pad_row = G_.new_tensor([0, 0, 0, 1])[None, None, None]
        pad_row = pad_row.expand(batch_size, 24, -1, -1)
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat([J, torch.zeros(batch_size, 24, 1).to(device)], dim=2).view(
            batch_size, 24, 4, 1)
        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = torch.matmul(self.weights,
                         G.permute(1, 0, 2, 3).contiguous().view(24, -1)).view(6890,
                                                                               batch_size,
                                                                               4,
                                                                               4).transpose(
            0, 1)
        rest_shape_h = torch.cat(
            [v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1)
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        return v

    def get_full_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 24, 3)
        """
        joints = torch.einsum('bik,ji->bjk', [vertices, self.J_regressor])
        return joints

    def get_leaf_joints(self, joints):
        leaf_indexes = [0, 7, 8, 12, 20, 21]
        return joints[:, leaf_indexes, :]


def get_smpl_vertices(trans: torch.Tensor,
                      poses: torch.Tensor,
                      shapes: torch.Tensor,
                      smpl: SMPL):
    vertices = smpl(poses, shapes)
    vertices += trans.unsqueeze(1)
    return vertices


def split_smpl_params(smpl_params: torch.Tensor):
    if smpl_params.size(-1) == 85:
        trans = smpl_params[..., :3].contiguous()
        poses = smpl_params[..., 3:3 + 72].contiguous()
        shapes = smpl_params[..., 3 + 72:].contiguous()
        return trans, poses, shapes
    else:
        poses = smpl_params[..., :72].contiguous()
        shapes = smpl_params[..., 72:].contiguous()
        return poses, shapes


colors = {
    # colorbline/print/copy safe:
    'light_blue': [244 / 255, 176 / 255, 132 / 255],
    'light_pink': [.9, .7, .7],  # This is used to do no-3d
}


class SMPLRenderer(object):
    def __init__(self,
                 face_path="/home/ljl/lidarcap/data/smpl_faces.npy"):
        self.faces = np.load(face_path)

    def __call__(self,
                 verts,
                 img,
                 do_alpha=False,
                 far=None,
                 near=None,
                 color_id=0,
                 img_size=None):
        """
        cam is 3D [f, px, py]
        """
        if img is not None:
            h, w = img.shape[:2]
        elif img_size is not None:
            h = img_size[0]
            w = img_size[1]
        else:
            h = self.h
            w = self.w

        use_cam = ProjectPoints(
            f=np.array([9.5632709662202160e+02, 9.5687763573729683e+02]),
            rt=np.zeros(3),
            t=np.zeros(3),
            k=np.array([-6.1100617222502205e-03, 3.0647823796371827e-02,
                        -3.3304524444662654e-04, -4.4038460096976607e-04,
                        -2.5974982760794661e-02]),
            c=np.array([9.6209910493679433e+02, 5.9026610775785059e+02]))

        if near is None:
            near = np.maximum(np.min(verts[:, 2]) - 25, 0.1)
        if far is None:
            far = np.maximum(np.max(verts[:, 2]) + 25, 25)

        imtmp = render_model(
            verts,
            self.faces,
            w,
            h,
            use_cam,
            do_alpha=do_alpha,
            img=img,
            far=far,
            near=near,
            color_id=color_id)

        return (imtmp * 255).astype('uint8')

    def rotated(self,
                verts,
                deg,
                cam=None,
                axis='y',
                img=None,
                do_alpha=True,
                far=None,
                near=None,
                color_id=0,
                img_size=None):
        import math
        if axis == 'y':
            around = cv2.Rodrigues(np.array([0, math.radians(deg), 0]))[0]
        elif axis == 'x':
            around = cv2.Rodrigues(np.array([math.radians(deg), 0, 0]))[0]
        else:
            around = cv2.Rodrigues(np.array([0, 0, math.radians(deg)]))[0]
        center = verts.mean(axis=0)
        new_v = np.dot((verts - center), around) + center

        return self.__call__(
            new_v,
            cam,
            img=img,
            do_alpha=do_alpha,
            far=far,
            near=near,
            img_size=img_size,
            color_id=color_id)


def _create_renderer(w=640,
                     h=480,
                     rt=np.zeros(3),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.5,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn


def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                   [-np.sin(angle), 0., np.cos(angle)]])
    return np.dot(points, ry)


def simple_renderer(rn,
                    verts,
                    faces,
                    yrot=np.radians(120),
                    color=colors['light_pink']):
    # Rendered model color
    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))
    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))

    return rn.r


def get_alpha(imtmp, bgval=1.):
    h, w = imtmp.shape[:2]
    alpha = (~np.all(imtmp == bgval, axis=2)).astype(imtmp.dtype)

    b_channel, g_channel, r_channel = cv2.split(imtmp)

    im_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha.astype(
        imtmp.dtype)))
    return im_RGBA


def append_alpha(imtmp):
    alpha = np.ones_like(imtmp[:, :, 0]).astype(imtmp.dtype)
    if np.issubdtype(imtmp.dtype, np.uint8):
        alpha = alpha * 255
    b_channel, g_channel, r_channel = cv2.split(imtmp)
    im_RGBA = cv2.merge((b_channel, g_channel, r_channel, alpha))
    return im_RGBA


def render_model(verts,
                 faces,
                 w,
                 h,
                 cam,
                 near=0.5,
                 far=25,
                 img=None,
                 do_alpha=False,
                 color_id=None):
    rn = _create_renderer(
        w=w, h=h, near=near, far=far, rt=cam.rt, t=cam.t, f=cam.f, c=cam.c)

    # Uses img as background, otherwise white background.
    if img is not None:
        rn.background_image = img / 255. if img.max() > 1 else img

    if color_id is None:
        color = colors['light_blue']
    else:
        color_list = list(colors.values())
        color = color_list[color_id % len(color_list)]

    imtmp = simple_renderer(rn, verts, faces, color=color)

    # If white bg, make transparent.
    if img is None and do_alpha:
        imtmp = get_alpha(imtmp)
    elif img is not None and do_alpha:
        imtmp = append_alpha(imtmp)

    return imtmp