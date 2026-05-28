# IWPC visualisation script for single-electron QmisID probabiliity 

from TMTrain_single import KappaLightning
from iwpc.utils import latest_ckpt
from bokeh.io import curdoc
from iwpc.visualise.bokeh_function_visualiser_2D import BokehFunctionVisualiser2D
import torch 

module = KappaLightning.load_from_checkpoint(latest_ckpt("/Users/albaburgosmondejar/QMISID/phi_training/kappa/version_1"))
curdoc().add_root(BokehFunctionVisualiser2D.visualise(
module,
initial_x_axis_scalar_ind=0,
initial_y_axis_scalar_ind=0,
initial_output_scalar_ind=0,

# label_font_size="36px",
# tick_font_size="26px",
selected_input_parameter_resolution = 600).root)