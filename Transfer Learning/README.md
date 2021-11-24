# Transfer Learning
This part is to be run with a PC, not the Raspberry Pi.

We recommend using Linux distribution to run the notebook, but windows is also acceptable.

# Using the notebook
First of all, you must have access to `jupyter notebook` or `jupyter lab`. You can read bout it [here](https://jupyter.org/index.html).

After getting `Jupyter` but before heading up to the `.ipynb` file, be sure you have intalled the object detection APi. To do so, follow according to your distribution.

## Windows

Open the `install_object_detection_API.sh` file with your prefered software.

copy and paste the instructions line by line inside a command prompt. It has to be inside the `Transfer\ Learning/` directory.

## Linux
In a terminal inside `Transfer\ Learning/` directory, run:

```
sudo bash Transfer\ Learning/install_object_detection_API.sh
```

After running the script, you must have the following output:

```
...
...
[ RUN      ] ModelBuilderTF2Test.test_unknown_meta_architecture
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_unknown_meta_architecture): 0.0s
I1124 15:05:42.600697 13488 test_util.py:2188] time(__main__.ModelBuilderTF2Test.test_unknown_meta_architecture): 0.0s
[       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_unknown_ssd_feature_extractor): 0.0s
I1124 15:05:42.601695 13488 test_util.py:2188] time(__main__.ModelBuilderTF2Test.test_unknown_ssd_feature_extractor): 0.0s
[       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 24 tests in 15.356s

OK (skipped=1)
```

If more then one test skipped, you may have some problems in the future. If that's the case, revisit the [documentation for the Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html).
