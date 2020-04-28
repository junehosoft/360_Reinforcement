# Adopted from code written by Yuting Yang, 2019
import numpy as np
import numpy
#import skimage.io
#import skimage
from matplotlib import pyplot
import time

def coord2ang(x, y, w, fov):
  """
  x, y are coordinages in image domain
  w is image width
  fov is camera field of view horizontally
  assuming camera has the same scale horizontally and vertically
  assuming phi, theta are all 0 at the center of the image (x=0, y=0)
  theta is horizontal angle
  phi is vertical angle
  """
  # r is distance from camera to the center of the image
  r = (w / 2.0) / np.tan(fov / 2.0)
  theta = np.arctan2(x, r)
  #phi = np.arctan(y / r)
  phi = np.arctan2(y, (r * r + x * x) ** 0.5)

  return theta, phi

def ang2coord(theta, phi, w, fov):
  """
  theta, phi are angles to image domain horizontally and vertically
  w is image width
  fov is camera field of view horizontally
  assuming camera has the same scale horizontally and vertically
  assuming phi = 0, theta = 0 projects to x = 0, y = 0
  """
  r = (w / 2.0) / np.tan(fov / 2.0)
  x = r * np.tan(theta)
  y = (r * r + x * x) ** 0.5 * np.tan(phi)
  #x[np.abs(theta) > 0.5 * np.pi] = np.inf
  return x, y

def ang2sphere_coord(theta, phi):
  """
  theta, phi are angles in image domain horizontally and vertically
  output Cartesian coordinates as if theta, phi are projected to a sphere of radius 1
  """
  x = np.cos(phi) * np.cos(theta)
  y = np.cos(phi) * np.sin(theta)
  z = np.sin(phi)
  return x, y, z

def sphere_coord2ang(x, y, z):
  """
  x, y, z are coordinates on a radius 1 sphere
  return angle coordinates at that point
  """
  phi = np.arcsin(z / (x * x + y * y + z * z) ** 0.5)
  theta = np.arctan2(y, x)
  return theta, phi

def roty(cam_v_ang):
  return numpy.array([[np.cos(cam_v_ang), 0.0, np.sin(cam_v_ang)],
                      [0.0, 1.0, 0.0],
                      [-np.sin(cam_v_ang), 0.0, np.cos(cam_v_ang)]])

def rotz(cam_h_ang):
  return numpy.array([[np.cos(cam_h_ang), -np.sin(cam_h_ang), 0.0],
                      [np.sin(cam_h_ang), np.cos(cam_h_ang), 0.0],
                      [0.0, 0.0, 1.0]])

def cam2glob(flatten_x, flatten_y, flatten_z, camera_h, camera_v):
  sinv = np.sin(camera_v)
  cosv = np.cos(camera_v)
  sinh = np.sin(camera_h)
  cosh = np.cos(camera_h)

  rotated_x = cosv * flatten_x + sinv * flatten_z
  rotated_y = flatten_y
  rotated_z = -sinv * flatten_x + cosv * flatten_z

  final_x = cosh * rotated_x - sinh * rotated_y
  final_y = sinh * rotated_x + cosh * rotated_y
  final_z = rotated_z
  return final_x, final_y, final_z

def glob2cam(flatten_x, flatten_y, flatten_z, camera_h, camera_v):
  sinv = np.sin(camera_v)
  cosv = np.cos(camera_v)
  sinh = np.sin(camera_h)
  cosh = np.cos(camera_h)

  rotated_x = cosh * flatten_x + sinh * flatten_y
  rotated_y = -sinh * flatten_x + cosh * flatten_y
  rotated_z = flatten_z

  final_x = cosv * rotated_x - sinv * rotated_z
  final_y = rotated_y
  final_z = sinv * rotated_x + cosv * rotated_z
  return final_x, final_y, final_z

def nfov2glob_ang(i, j, nfov_w, nfov_h, fov, cam_h, cam_v):
  x = i.astype('f') - float(nfov_w) / 2.0
  y = j.astype('f') - float(nfov_h) / 2.0
  theta, phi = coord2ang(x, y, nfov_w, fov)

  sphere_x, sphere_y, sphere_z = ang2sphere_coord(theta, phi)

  final_x, final_y, final_z = cam2glob(sphere_x, sphere_y, sphere_z, cam_h, cam_v)

  glob_theta, glob_phi = sphere_coord2ang(final_x, final_y, final_z)
  return glob_theta, glob_phi

def glob_ang2nfov(glob_theta, glob_phi, nfov_w, nfov_h, fov, cam_h, cam_v):
  x, y, z = ang2sphere_coord(glob_theta, glob_phi)
  xx, yy, zz = glob2cam(x, y, z, cam_h, cam_v)
  cam_theta, cam_phi = sphere_coord2ang(xx, yy, zz)
  x, y = ang2coord(cam_theta, cam_phi, nfov_w, fov)
  i = x + float(nfov_w) / 2.0
  j = y + float(nfov_h) / 2.0
  i[np.abs(cam_theta) > 0.5 * np.pi] = -1
  j[np.abs(cam_theta) > 0.5 * np.pi] = -1
  return i, j

def pix_mod_ind(img, x, y, mask=None):
  ans = img[y % img.shape[0], x % img.shape[1], :]
  if mask is not None:
    mask[y % img.shape[0], x % img.shape[1]] = 1.0
  return ans

def get_pix_val_nfov2equi(glob_theta, glob_phi, img, par_set=[None]*9):
  equi_w = img.shape[1]
  equi_h = img.shape[0]

  equi_x = (glob_theta / (2.0 * np.pi)) * equi_w + equi_w / 2.0
  equi_y = (glob_phi / np.pi) * equi_h + equi_h / 2.0

  low_x = np.floor(equi_x).astype('i')
  hi_x = np.ceil(equi_x).astype('i')
  low_y = np.floor(equi_y).astype('i')
  hi_y = np.ceil(equi_y).astype('i')

  #this is used to get weights of pixels kind of
  x_pct = np.expand_dims(equi_x.astype('f') - low_x.astype('f'), axis=2)
  y_pct = np.expand_dims(equi_y.astype('f') - low_y.astype('f'), axis=2)
  #print('x_pct:', x_pct.shape, x_pct)
  #print('y_pct:', y_pct.shape, y_pct)
  #ans = pix_mod_ind(img, low_x, low_y) + pix_mod_ind(img, hi_x, low_y)

  mask = np.zeros((equi_h, equi_w))

  ans = pix_mod_ind(img, low_x, low_y, mask).astype('f') * (1.0 - x_pct) * (1.0 - y_pct) + \
        pix_mod_ind(img, low_x, hi_y, mask).astype('f') * (1.0 - x_pct) * y_pct + \
        pix_mod_ind(img, hi_x, low_y, mask).astype('f') * x_pct * (1.0 - y_pct) + \
        pix_mod_ind(img, hi_x, hi_y, mask).astype('f') * x_pct * y_pct

  par_set[0] = low_x % img.shape[1]
  par_set[1] = hi_x % img.shape[1]
  par_set[2] = low_y % img.shape[0]
  par_set[3] = hi_y % img.shape[0]
  par_set[4] = (1.0 - x_pct) * (1.0 - y_pct)
  par_set[5] = (1.0 - x_pct) * y_pct
  par_set[6] = x_pct * (1.0 - y_pct)
  par_set[7] = x_pct * y_pct
  par_set[8] = mask

  return ans.astype('uint8'), mask

def pix_valid_ind(img, x, y, mask=None):
  valid_mask = (x >= 0) * (x < img.shape[1]) * (y >= 0) * (y < img.shape[0])
  #invalid_mask = (1.0 - valid_mask).astype('bool')
  #x[invalid_mask] = 0
  #y[invalid_mask] = 0
  #ans = np.expand_dims(valid_mask, 2) * img[y, x, :]
  ans = np.zeros((valid_mask.shape[0], valid_mask.shape[1], img.shape[2]))
  ans[valid_mask, :] = img[y[valid_mask], x[valid_mask], :]
  if mask is not None:
    mask[valid_mask] = 1.0
  return ans, np.expand_dims(valid_mask.astype('f'), axis=2)

def get_pix_val_equi2nfov(i, j, img, par_set=[None]*14):
  mask = np.zeros((i.shape[0], i.shape[1]))

  ans_ll, mask_ll = pix_valid_ind(img, np.floor(i).astype('i'), np.floor(j).astype('i'), mask)
  ans_lh, mask_lh = pix_valid_ind(img, np.floor(i).astype('i'), np.ceil(j).astype('i'), mask)
  ans_hl, mask_hl = pix_valid_ind(img, np.ceil(i).astype('i'), np.floor(j).astype('i'), mask)
  ans_hh, mask_hh = pix_valid_ind(img, np.ceil(i).astype('i'), np.ceil(j).astype('i'), mask)

  x_pct = np.expand_dims(i.astype('f') - np.floor(i).astype('f'), axis=2)
  y_pct = np.expand_dims(j.astype('f') - np.floor(j).astype('f'), axis=2)

  ans_sum = ans_ll.astype('f') * (1.0 - x_pct) * (1.0 - y_pct) + \
            ans_lh.astype('f') * (1.0 - x_pct) * y_pct + \
            ans_hl.astype('f') * x_pct * (1.0 - y_pct) + \
            ans_hh.astype('f') * x_pct * y_pct

  weight_sum = mask_ll * (1.0 - x_pct) * (1.0 - y_pct) + \
               mask_lh * (1.0 - x_pct) * y_pct + \
               mask_hl * x_pct * (1.0 - y_pct) + \
               mask_hh * x_pct * y_pct
  par_set[0] = np.floor(i).astype('i')
  par_set[1] = np.ceil(i).astype('i')
  par_set[2] = np.floor(j).astype('i')
  par_set[3] = np.ceil(j).astype('i')
  par_set[4] = (1.0 - x_pct) * (1.0 - y_pct)
  par_set[5] = (1.0 - x_pct) * y_pct
  par_set[6] = x_pct * (1.0 - y_pct)
  par_set[7] = x_pct * y_pct
  par_set[8] = np.squeeze(mask_ll).astype('bool')
  par_set[9] = np.squeeze(mask_lh).astype('bool')
  par_set[10] = np.squeeze(mask_hl).astype('bool')
  par_set[11] = np.squeeze(mask_hh).astype('bool')
  par_set[12] = weight_sum
  par_set[13] = mask
  return (ans_sum / weight_sum).astype('uint8'), mask

def get_nfov_img(w, h, fov, cam_h, cam_v, img, par_set=None):
  if par_set is None:
    par_set = [None] * 9
    #print(w, h)
    i, j = numpy.meshgrid(range(w), range(h))
    glob_theta, glob_phi = nfov2glob_ang(i, j, w, h, fov, cam_h, cam_v)
    #print("glob_theta:", glob_theta.shape, glob_theta)
    #print("glob_phi:", glob_phi.shape, glob_phi)
    ans, mask = get_pix_val_nfov2equi(glob_theta, glob_phi, img, par_set=par_set)
  else:
    #print("using precomputed parameters")
    ans = img[par_set[2], par_set[0]].astype('f') * par_set[4] + \
          img[par_set[3], par_set[0]].astype('f') * par_set[5] + \
          img[par_set[2], par_set[1]].astype('f') * par_set[6] + \
          img[par_set[3], par_set[1]].astype('f') * par_set[7]
    mask = par_set[8]
  #pyplot.figure()
  #pyplot.imshow(ans)
  #pyplot.figure()
  #pyplot.imshow(mask)
  #pyplot.figure()
  #pyplot.imshow(img)
  return ans.astype('uint8'), par_set

def get_equi_img(equi_w, equi_h, n_fov, cam_h, cam_v, img, par_set=None):
  if par_set is None:
    par_set = [None] * 14
    i, j = numpy.meshgrid(range(equi_w), range(equi_h))
    glob_theta = (i - equi_w / 2.0) / equi_w * (2.0 * np.pi)
    glob_phi = (j - equi_h / 2.0) / equi_h * (np.pi)
    ii, jj = glob_ang2nfov(glob_theta, glob_phi, img.shape[1], img.shape[0], n_fov, cam_h, cam_v)
    ans, mask = get_pix_val_equi2nfov(ii, jj, img, par_set=par_set)
  else:
    #print("using precomputed parameters")
    ans = numpy.zeros((par_set[0].shape[0], par_set[0].shape[1], img.shape[2]))
    low_i = par_set[0].astype('i')
    hi_i = par_set[1].astype('i')
    low_j = par_set[2].astype('i')
    hi_j = par_set[3].astype('i')
    w0 = par_set[4]
    w1 = par_set[5]
    w2 = par_set[6]
    w3 = par_set[7]
    m0 = par_set[8]
    m1 = par_set[9]
    m2 = par_set[10]
    m3 = par_set[11]
    ans[m0] += img[low_j[m0], low_i[m0], :].astype('f') * w0[m0]
    ans[m1] += img[hi_j[m1], low_i[m1], :].astype('f') * w1[m1]
    ans[m2] += img[low_j[m2], hi_i[m2], :].astype('f') * w2[m2]
    ans[m3] += img[hi_j[m3], hi_i[m3], :].astype('f') * w3[m3]
    ans /= par_set[12]
    mask = par_set[13]
  #pyplot.figure()
  #pyplot.imshow(ans.astype('uint8'))
  #pyplot.figure()
  #pyplot.imshow(mask)
  #pyplot.figure()
  #pyplot.imshow(img)
  return ans.astype('uint8'), par_set

'''if True:
  import cv2
  cap = cv2.VideoCapture("/content/drive/My Drive/research/code/360-to-salient-video/videos/Stimuli/1_PortoRiverside.mp4")
  _, img = cap.read()
  cap.release()

  t1 = time.time()
  img_nfov, par_set = get_nfov_img(960, 640, np.pi / 2.0, np.pi, 0.5, img)
  t2 = time.time()
  #import skimage.io
  #skimage.io.imsave('test_nfov.png', img_nfov)
  pyplot.figure()
  pyplot.imshow(img_nfov)
  pyplot.figure()
  pyplot.imshow(par_set[-1])

  t3 = time.time()
  img2, _ = get_nfov_img(960, 640, np.pi / 2.0, np.pi, 0.5, img, par_set=par_set)
  t4 = time.time()
  pyplot.figure()
  pyplot.imshow(img2)

  print(t2 - t1)
  print(t4 - t3)

  t5 = time.time()
  img_back, par_set = get_equi_img(img.shape[1], img.shape[0], np.pi / 2.0, np.pi, 0.5, img_nfov)
  t6 = time.time()
  img_back2, _ = get_equi_img(img.shape[1], img.shape[0], np.pi / 2.0, np.pi, 0.5, img_nfov, par_set)
  t7 = time.time()
  print(t6 - t5)
  print(t7 - t6)
  #skimage.io.imsave('test.png', img_back)

  def get_dodecahedron_angs():
    phi = (5 ** 0.5 + 1.0) / 2.0
    vertices = numpy.array([[1.0, 1.0, 1.0],
                            [1.0, 1.0, -1.0],
                            [1.0, -1.0, 1.0],
                            [1.0, -1.0, -1.0],
                            [-1.0, 1.0, 1.0],
                            [-1.0, 1.0, -1.0],
                            [-1.0, -1.0, 1.0],
                            [-1.0, -1.0, -1.0],
                            [0.0, phi, 1.0 / phi],
                            [0.0, phi, -1.0 / phi],
                            [0.0, -phi, 1.0 / phi],
                            [0.0, -phi, -1.0 / phi],
                            [1.0 / phi, 0.0, phi],
                            [1.0 / phi, 0.0, -phi],
                            [-1.0 / phi, 0.0, phi],
                            [-1.0 / phi, 0.0, -phi],
                            [phi, 1.0 / phi, 0.0],
                            [phi, -1.0 / phi, 0.0],
                            [-phi, 1.0 / phi, 0.0],
                            [-phi, -1.0 / phi, 0.0]])

    glob_theta, glob_phi = sphere_coord2ang(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    return glob_theta, glob_phi

def get_equi2dodecaheron_pars(w, h, fov, img, equi_w, equi_h):
  glob_theta, glob_phi = get_dodecahedron_angs()
  parameters = []
  parameters_back = []
  for i in range(glob_theta.shape[0]):
    print(i)
    ans, par_set = get_nfov_img(w, h, fov, glob_theta[i], glob_phi[i], img)
    #print('forward parameters get')
    ans, par_set_back = get_equi_img(equi_w, equi_h, fov, glob_theta[i], glob_phi[i], ans)
    #print('backward parameters get')
    parameters.append(par_set)
    parameters_back.append(par_set_back)
  return parameters, parameters_back

def equi2dodecahedron(img, parameters):
  #glob_theta, glob_phi = get_dodecahedron_angs()
  #assert len(parameters) == glob_theta.shape[0]

  ans = []
  for i in range(len(parameters)):
    current_img, _ = get_nfov_img(0, 0, 0, 0, 0, img, parameters[i])
    ans.append(current_img)
  return ans

def nfov2dodecahedron(imgs, parameters):
  ans = numpy.zeros((parameters[0][0].shape[0], parameters[0][0].shape[1], imgs[0].shape[2]))
  mask_sum = numpy.zeros(parameters[0][12].shape)
  for i in range(len(parameters)):
    current, _ = get_equi_img(0, 0, 0, 0, 0, imgs[i], parameters[i])
    ans += current.astype('f') * parameters[i][12].astype('f')
    mask_sum += parameters[i][12].astype('f')
  ans /= mask_sum
  return ans.astype('uint8')

#glob_theta, glob_phi = get_dodecahedron_angs()
if True:
  import cv2
  cap = cv2.VideoCapture("/content/drive/My Drive/research/code/360-to-salient-video/videos/Stimuli/1_PortoRiverside.mp4")
  _, img = cap.read()
  cap.release()
  img = cv2.resize(img, (960, 480))

  fov = np.pi / 2.0

  t1 = time.time()
  #parameters, parameters_back = get_equi2dodecaheron_pars(960, 640, fov, img, 960, 480)
  parameters_forward, parameters_back = get_equi2dodecaheron_pars(796, 448, fov, img, 960, 480)
  t2 = time.time()
ans = equi2dodecahedron(img, parameters_forward)
t3 = time.time()
ans_back = nfov2dodecahedron(ans, parameters_back)
t4 = time.time()

for i in range(len(ans)):
  pyplot.figure()
  pyplot.subplot(121)
  pyplot.imshow(ans[i])
  pyplot.subplot(122)
  pyplot.imshow(parameters_forward[i][-1])

pyplot.figure()
pyplot.imshow(ans_back)
pyplot.figure()
pyplot.imshow(img)
print(t2 - t1)
print(t3 - t2)
print(t4 - t3)

t4 = time.time()'''
