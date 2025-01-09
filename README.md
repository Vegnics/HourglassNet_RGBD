# Customized Stacked Hourglass Network (HGNet) for in-bed pose estimation using RGB-D images
<p style="text-align:center;">
<a href="https://github.com/Vegnics/HourglassNet_RGBD" alt="Python"><img src="https://img.shields.io/badge/python-3.10-blue" alt="Python Version" /></a>
<a href="https://github.com/Vegnics/HourglassNet_RGBD/releases" alt="Releases"><img src="https://img.shields.io/github/v/release/Vegnics/HourglassNet_RGBD" alt="Latest Version" /></a>
<a href="https://github.com/Vegnics/HourglassNet_RGBD/blob/main/LICENSE" alt="Licence"><img src="https://img.shields.io/github/license/Vegnics/HourglassNet_RGBD" alt="Licence" /></a>
</p>
<p style="text-align:center;">
<a href="https://github.com/Vegnics/HourglassNet_RGBD/commits" alt="Stars"><img src="https://img.shields.io/github/commit-activity/m/Vegnics/HourglassNet_RGBD" alt="Commit Activity" /></a>
<a href="https://github.com/Vegnics/HourglassNet_RGBD" alt="Repo Size"><img src="https://img.shields.io/github/repo-size/Vegnics/HourglassNet_RGBD" alt="Repo Size" /></a>
<a href="https://github.com/Vegnics/HourglassNet_RGBD" alt="Issues"><img src="https://img.shields.io/github/issues/Vegnics/HourglassNet_RGBD" alt="Issues" /></a>
<a href="https://github.com/Vegnics/HourglassNet_RGBD" alt="Pull Requests"><img src="https://img.shields.io/github/issues-pr/Vegnics/HourglassNet_RGBD" alt="Pull Requests" /></a>
<a href="https://github.com/Vegnics/HourglassNet_RGBD" alt="Downloads"><img src="https://img.shields.io/github/downloads/Vegnics/HourglassNet_RGBD/total" alt="Downloads" /></a>
</p>
<p style="text-align:center;">
<a href="https://github.com/Vegnics/HourglassNet_RGBDactions" alt="Build Status"><img src="https://github.com/Vegnics/HourglassNet_RGBD/actions/workflows/python-release.yaml/badge.svg" alt="Build Status" /></a>
<a href="https://github.com/Vegnics/HourglassNet_RGBD/actions" alt="Test Status"><img src="https://github.com/Vegnics/HourglassNet_RGBD/actions/workflows/python-test.yaml/badge.svg" alt="Test Status" /></a>
<a href="https://github.com/Vegnics/HourglassNet_RGBD/actions" alt="Publish Status"><img src="https://github.com/Vegnics/HourglassNet_RGBD/actions/workflows/python-publish.yaml/badge.svg" alt="Publish Status" /></a>
</p>
<p style="text-align:center;">
<a href="https://github.com/Vegnics/HourglassNet_RGBD" alt="Tests"><img src="./reports/tests-badge.svg" alt="Tests"/></a>
<a href="https://github.com/Vegnics/HourglassNet_RGBD" alt="Coverage"><img src="./reports/coverage-badge.svg" alt="Coverage"/></a>
<a href="https://github.com/Vegnics/HourglassNet_RGBD" alt="Flake8"><img src="./reports/flake8-badge.svg" alt="Flake8"/></a>
</p>
<p style="text-align:center;">
<a href="https://github.com/Vegnics/HourglassNet_RGBD/stargazers" alt="Stars"><img src="https://img.shields.io/github/stars/Vegnics/HourglassNet_RGBD?style=social" alt="Stars" /></a>
<a href="https://github.com/Vegnics/HourglassNet_RGBD" alt="Forks"><img src="https://img.shields.io/github/forks/Vegnics/HourglassNet_RGBD?style=social" alt="Forks" /></a>
<a href="https://github.com/Vegnics/HourglassNet_RGBD/watchers" alt="Watchers"><img src="https://img.shields.io/github/watchers/Vegnics/HourglassNet_RGBD?style=social" alt="Watchers" /></a>
</p>

This repository is a reimplementation of _A.Newell et Al_, [_**Stacked Hourglass Network for Human Pose Estimation**_](https://arxiv.org/abs/1603.06937) using **TensorFlow 2**. Modifications to the main blocks of the HGNet are included in this repository.

Collaborative project with NDHU _(National Dong Hwa University)_. 

This project was based on **Walid Benbihi's** implementation. The models can be trained on [**MPII Human Pose Dataset**](http://human-pose.mpi-inf.mpg.de/), [**SLP Multimodal in-bed pose estimation dataset**](https://ostadabbas.sites.northeastern.edu/slp-dataset-for-multimodal-in-bed-pose-estimation-3/).

Currrently, the framework for training in-bed pose estimation models regards **depth** and **RGB-D** modalities. Further documentation is still under development.  
