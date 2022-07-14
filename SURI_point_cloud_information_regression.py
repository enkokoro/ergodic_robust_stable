import numpy as np
import sklearn.gaussian_process as gp
import pyvista as pv
import random
"""
POINT CLOUD SPECIFIC INPUTS

MODEL NAME
POINT CLOUD
MEASUREMENT
"""
model_name = "cube"
def model_point_cloud(): # return numpy array dimenstion (n_points, 3)
    return cube_point_cloud(10) 
def train_data(): # return (numpy array dimenstion (n_points, 3), scalar array dimension (n_points,)
    return cube_train_data()
def measurement(data):
    return np.max(data, axis=1)

# model_name = "sphere"
# model_name = "cone"
# def model_point_cloud(): # return numpy array dimenstion (n_points, 3)
#     # return pv.Sphere().points
#     return pv.Cone().points
# def train_data(): # return (numpy array dimenstion (n_points, 3), scalar array dimension (n_points,)
#     pc = model_point_cloud()
#     num_samples = min(10, len(pc)//2)
#     idx = random.sample(range(0, len(pc)), num_samples)
#     data_tr = pc[idx,:]
#     return data_tr, measurement(data_tr)
# def measurement(data):
#     return np.max(data, axis=1)

"""
FUNCTION REGRESSION SPECIFIC INPUTS
    technically any function regression model works as long as it obeys the following
    
    model.fit(data_tr, val_tr)
    val_te, std_te = model.predict(data_te, return_std=True)
"""
# GAUSSIAN PROCESS REGRESSION MODEL
kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)

###############################################################################
def cube_point_cloud(density=10):
    side_x, side_y = np.meshgrid(np.linspace(0, 1, density), np.linspace(0, 1, density))
    side_x = side_x.ravel()
    side_y = side_y.ravel()
    
    x = np.hstack([side_x]*4+[0]*density*density+[1]*density*density)
    y = np.hstack([side_y]+[1]*density*density+[side_y]+[0]*density*density+[side_y]*2)
    z = np.hstack([0]*density*density+[side_y]+[1]*density*density+[side_y]+[side_x]*2)
    return np.column_stack([x,y,z])
def cube_train_data():
    x_tr, y_tr, z_tr = [0, 0.25, 0.5, 0, 0, 0, 0, 1],[0, 0, 0, 0.25, 0.5, 0, 0, 1],[0, 0, 0, 0, 0, 0.25, 0.5, 1]# cube_point_cloud(5)
    data_tr = np.column_stack([x_tr, y_tr, z_tr]) # (n_points, 3)
    val_tr = measurement(data_tr)
    return data_tr, val_tr

###############################################################################
def pyvista_gif(plt, model_name, filename, shift):
    print(filename)
    plt.add_title(filename, font_size=12)
    plt.show(auto_close=False)
    path = plt.generate_orbital_path(n_points=36, shift=shift)
    plt.open_gif(f"gaussianprocess/{model_name}_{filename}.gif")
    plt.orbit_on_path(path, write_frames=True)
    plt.close()

# object point cloud
data_te = model_point_cloud() 
cloud_te = pv.wrap(data_te)
volume_te = cloud_te.delaunay_3d()
plt = pv.Plotter(off_screen=True)
plt.add_mesh(volume_te)
pyvista_gif(plt, model_name, "model", volume_te.length)

# plot training data
data_tr, val_tr = train_data()
cloud_tr = pv.PolyData(data_tr)
plt = pv.Plotter(off_screen=True)
plt.add_points(cloud_tr, scalars=val_tr, cmap='viridis', render_points_as_spheres=True, point_size=20.0)
plt.add_mesh(volume_te, opacity=0.75)
pyvista_gif(plt, model_name, "train_on_model", volume_te.length)

# fit model
model.fit(data_tr, val_tr)

# predicted values on model
val_te, std_te = model.predict(data_te, return_std=True)
plt = pv.Plotter(off_screen=True)
plt.add_points(cloud_tr, scalars=val_tr, cmap='viridis', render_points_as_spheres=True, point_size=20.0)
plt.add_mesh(volume_te, scalars=val_te, cmap='viridis')
pyvista_gif(plt, model_name, "test_on_model", volume_te.length)

# ground truth for comparison
val_gt = measurement(data_te)
plt = pv.Plotter(off_screen=True)
plt.add_mesh(volume_te, scalars=val_gt, cmap='viridis')
pyvista_gif(plt, model_name, "ground_truth_on_model", volume_te.length)

# error map
error = abs(val_gt - val_te)
plt = pv.Plotter(off_screen=True)
plt.add_points(cloud_tr, color="white", render_points_as_spheres=True, point_size=20.0)
plt.add_mesh(volume_te, scalars=error, cmap='binary')
pyvista_gif(plt, model_name, "error_on_model", volume_te.length)

# uncertainty on model
plt = pv.Plotter(off_screen=True)
plt.add_points(cloud_tr, color="white", render_points_as_spheres=True, point_size=20.0)
plt.add_mesh(volume_te, scalars=std_te, cmap='binary')
pyvista_gif(plt, model_name, "uncertainty_on_model", volume_te.length)


# iterative

for _i in range(len(data_tr)):
    i = _i + 1
    _data_tr = data_tr[:i,:]
    _val_tr = val_tr[:i]
    _cloud_tr = pv.PolyData(_data_tr)
    model.fit(_data_tr, _val_tr)
    _val_te, _std_te = model.predict(data_te, return_std=True)

    if _i == 0:
        info_clim = [np.min(val_gt), np.max(val_gt)]
        print("info: ", info_clim)
        uncertainty_clim = [0, np.max(abs(_std_te))]
        print("uncertainty: ", uncertainty_clim)
        error_clim = [0, np.max(abs(val_gt-_val_te))]
        print("error: ", error_clim)
    
    info_map = pv.Plotter(off_screen=True)
    info_map.add_title("Problematic Map", font_size=12)
    info_map.show(auto_close=False)
    info_map.open_gif(f"gaussianprocess/{model_name}_problem_map"+"_{:02d}.gif".format(i))

    info_map.add_points(_cloud_tr, scalars=_val_tr, cmap='viridis', render_points_as_spheres=True, point_size=20.0)
    info_map.add_mesh(volume_te, scalars=_val_te, cmap='viridis', clim=info_clim)
    info_path = info_map.generate_orbital_path(n_points=36, shift=volume_te.length)
    info_map.orbit_on_path(info_path, write_frames=True)

    info_map.close()

    uncertainty_map = pv.Plotter(off_screen=True)
    uncertainty_map.add_title("Uncertainty Map", font_size=12)
    uncertainty_map.show(auto_close=False)
    uncertainty_map.open_gif(f"gaussianprocess/{model_name}_uncertainty_map"+"_{:02d}.gif".format(i))

    uncertainty_map.add_points(_cloud_tr, color="white", render_points_as_spheres=True, point_size=20.0)
    uncertainty_map.add_mesh(volume_te, scalars=_std_te, cmap='binary', clim=uncertainty_clim)
    uncertainty_path = uncertainty_map.generate_orbital_path(n_points=36, shift=volume_te.length)
    uncertainty_map.orbit_on_path(uncertainty_path, write_frames=True)

    uncertainty_map.close()

    error_map = pv.Plotter(off_screen=True)
    error_map.add_title("Error Map", font_size=12)
    error_map.show(auto_close=False)
    error_map.open_gif(f"gaussianprocess/{model_name}_error_map"+"_{:02d}.gif".format(i))

    error_map.add_points(_cloud_tr, color="white", render_points_as_spheres=True, point_size=20.0)
    error_map.add_mesh(volume_te, scalars=abs(val_gt-_val_te), cmap='binary', clim=error_clim)
    error_path = error_map.generate_orbital_path(n_points=36, shift=volume_te.length)
    error_map.orbit_on_path(error_path, write_frames=True)

    error_map.close()