Hello, I am trying to segment cell images with Omnipose using two different approaches and I am running into different errors for both methods. I am following the example code on the [Omnipose documentation](https://omnipose.readthedocs.io/examples/mono_channel_3D.html) for 3D segmentation and have even tested with the [same file](https://github.com/kevinjohncutler/omnipose/tree/main/docs/test_files_3D) they used.

I suspect that the reason for my errors could be related to some issue with the installation of the Omnipose package or its dependencies because I have set it up on a virtual environment on the Narval cluster and some packages are from compute Canada while some are not (were not available on compute Canada). However, I am not sure if this is truly the case.

Throughout this post, I have linked to a [repo I made](https://github.com/mrdandelion6/omnipose_errors) containing the errors and the code producing them. 

Before I describe the errors and the code procuding them, [here is the result](https://github.com/mrdandelion6/omnipose_errors/blob/main/pip_list.txt) of a `pip list` in my environment:

## Method One: Using 3D plant_omni Model

When I attempt to segment 3D images using Omnipose's `plant_omni` 3D model, I get a tensor overflow error. Here is the model initialization code:
```python
from cellpose_omni import models

# need to provide omni = true or else the mode will net_avg
# omni doesnt use net_avg
# if omni is false, we will need net_avg and use 4 models for that

model = models.CellposeModel(gpu=True,
                             model_type="plant_omni",
                             net_avg=False,
                             dim=3, # model was trained on 2D slices
                             nchan=1,
                             diam_mean=40,
                             nclasses=3) # flow + dist + boundary
```

Here is the call to eval which is producing the error:
```python
torch.cuda.empty_cache()

masks_om, flows_om = [[]]*nimg,[[]]*nimg

for k in range(nimg):
    # imgs[k] = torch.from_numpy(imgs[k]).to(device) # attempt to try and fix the tensor overload bug
    masks_om[k], flows_om[k], _ = model.eval(imgs[k],
                                             channels=None,
                                             rescale=None,
                                             mask_threshold=-5,
                                             net_avg=False,
                                             transparency=True,
                                             flow_threshold=0,
                                             omni=True,
                                             resample=False,
                                             verbose=1,
                                             diam_threshold=55,
                                             cluster=False,
                                             tile=True,
                                             compute_masks=1,
                                             flow_factor=10) 
```

You can see the [full code here](https://github.com/mrdandelion6/omnipose_errors/blob/main/tensor_overflow_error/tensor_error_code.ipynb) but it is the same as what is on the Omnipose documentation with the exception of the file path.

Here is part of the error message I am getting from the above code:
```
RuntimeError: Expected output.numel() <= std::numeric_limits<int32_t>::max() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
```

Here is a link to the [full error message]().


