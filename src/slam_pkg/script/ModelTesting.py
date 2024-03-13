from sklearn.gaussian_process import GaussianProcessRegressor
import joblib


gp_loaded = joblib.load('/home/cocokayya18/Spezialisierung-1/src/slam_pkg/myMLmodel/GP_model.joblib')

x_predict = [[-2.000019313172563],[-0.4999996481424659],[-0.00021734587417554066],[8.57819098239314e-08],[6.18047277926345e-07],[9.128718510181314e-06]]

y_predict = gp_loaded.predict(x_predict, return_std=True)

print(y_predict)

print(gp_loaded.get_params())

print("GT_X: -2.2288948464677105e-09, GT_Y: 1.6343130682106022e-08, GT_Yaw: 1.4877646695738464e-05")