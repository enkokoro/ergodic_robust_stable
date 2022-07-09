import numpy as np
import sklearn.gaussian_process as gp
import pyvista as pv

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
    x_tr, y_tr, z_tr = [0, 0.25, 0.5, 0, 0, 0, 0, 1],[0, 0, 0, 0.25, 0.5, 0, 0, 1],[0, 0, 0, 0, 0, 0.25, 0.5, 1]# cube_point_cloud(5)
    data_tr = np.column_stack([x_tr, y_tr, z_tr]) # (n_points, 3)
    val_tr = measurement(data_tr)
    return data_tr, val_tr
def measurement(data):
    return np.max(data, axis=1)

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
def cube_point_cloud(density):
    side_x, side_y = np.meshgrid(np.linspace(0, 1, density), np.linspace(0, 1, density))
    side_x = side_x.ravel()
    side_y = side_y.ravel()
    
    x = np.hstack([side_x]*4+[0]*density*density+[1]*density*density)
    y = np.hstack([side_y]+[1]*density*density+[side_y]+[0]*density*density+[side_y]*2)
    z = np.hstack([0]*density*density+[side_y]+[1]*density*density+[side_y]+[side_x]*2)
    return np.column_stack([x,y,z])

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
plt = pv.Plotter(off_screen=True)
plt.add_mesh(volume_te, scalars=measurement(data_te), cmap='viridis')
pyvista_gif(plt, model_name, "ground_truth_on_model", volume_te.length)

# uncertainty on model
plt = pv.Plotter(off_screen=True)
plt.add_points(cloud_tr, color="white", render_points_as_spheres=True, point_size=20.0)
plt.add_mesh(volume_te, scalars=std_te, cmap='binary')
pyvista_gif(plt, model_name, "uncertainty_on_model", volume_te.length)
