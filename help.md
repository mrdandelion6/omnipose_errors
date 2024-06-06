Hello, I am trying to segment cell images with Omnipose using two different approaches and I am running into different errors for both methods. I am following the example code on the [Omnipose documentation](https://omnipose.readthedocs.io/examples/mono_channel_3D.html) for 3D segmentation and have even tested with the [same file](https://github.com/kevinjohncutler/omnipose/tree/main/docs/test_files_3D) they used.

I suspect that the reason for my errors could be related to some issue with the installation of the Omnipose package or its dependencies because I have set it up on a virtual environment on the Narval cluster and some packages are from compute Canada while some are not (were not available on compute Canada). However, I am not sure if this is truly the case.

Throughout this post, I have linked to a [repo I made](https://github.com/mrdandelion6/omnipose_errors) containing the errors and the code producing them. 

Before I describe the errors and the code procuding them, [here is the result](https://github.com/mrdandelion6/omnipose_errors/blob/main/pip_list.txt) of a `pip list` in my environment:

## Method One: Using 3D plant_omni Model

When I attempt to segment 3D images using Omnipose's `plant_omni` 3D model, I get a tensor overflow error. Here is the model initialization code:
```python

```

Here is the call to eval which is producing the error:
```python

```

You can see the [full code here]() but it is the same as what is on the Omnipose documentation with the exception of the file path.

Here is part of the error message I am getting from the above code:
```
RuntimeError: Expected output.numel() <= std::numeric_limits<int32_t>::max() to be true, but got false.  (Could this error message be improved?  If so, please report an enhancement request to PyTorch.)
```

Here is a link to the [full error message]().


