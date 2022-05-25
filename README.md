<!-- PROJECT SHIELDS -->

[![MIT License][license-shield]][license-url]
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ning-yu-kao)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kaoningyupage/)
[![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:kaoningyu@gmail.com)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)]()





<!-- PROJECT LOGO -->
<br />
<div align="center">
  <img src="https://github.com/ning-yu-kao/pinns-implements/blob/main/pinn.gif">
  <h1 align="center">Physics-informed neural networks (PINNs) Implementations: Teach Neural Network How to Learn Physics.</h1>
  <p>In the world of physics, it's much more complex than you might think, even if you have all data.
    Since recent growth of Deep Learning implementation, we can combine data-driven information and physics information in order to predict
    complex physical mechanism. This project showcase how to play around Physics-informed neural networks (PINNs) with some example.</p>
</div>

<!-- TABLE OF CONTENTS -->
<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#physics-informed-neural-networks">PINNs</a>
    </li>  
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#references">References</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


## Physics-informed Neural Networks

Let's briefly recap what's PINNs! 

<!-- ABOUT THE PROJECT -->
## About The Project

In this project, we use a 1D Heat Equation as an example to demostrate PINNs implementation. Once we finish training, we'll visualize the equation as 3D surface plot to compare with analytical solution.


### Built With

This is major frameworks/libraries used in this project.

* Python
* [Pytorch](https://pytorch.org/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Please follow the instruction below to set up your environment.

### Installation


1. Install packages

  * **PyTorch**
  
    > Here, we use PyTorch to build our neural network. Furthermore, since it's flexibility, it's much easier to implement PINNs.
    
    1. Visit [PyTorch website](https://pytorch.org/get-started/locally/) to install PyTorch first
    
    2. Select your OS/PyTorch/CUDA combination and install, for me it's:


        ```bash
        conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
        ```
    
    For more PyTorch detail, please check out [PyTorch](https://pytorch.org/) official website.

    

  * **Weights & Biases**
  
    <img src="https://drive.google.com/uc?export=view&id=1VdTXrUtR2ero3XCN1v4QLuCcw23KZFa5" width="300" height="50">
    
    > Weights & Biases is the MLOps platform for developers to build and monitor their model status.

    1. You can simply install it by:
    
        ```bash
        pip install wandb
        ```
    
    2. Sign up an account in [W&B website](https://wandb.ai/site). 
    
    3. Test this code with your [API key](https://wandb.ai/authorize).
    
        ```python
        wandb.login(anonymous="must", key='Your Key')
        print("Successfully Logged In!")
        ```
        
     You guys can view my output dashboard of a sigle run at this [link](https://wandb.ai/kaoningyu/gcn-ppi/reports/Predict-PPI-network-using-GCN--VmlldzoyMDIyNDgw?accessToken=0fgywscyq72iiivoyiq1lunofjhvkdqbejovoug456owxpjkax4ize40ug7wtwtr).
     
<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


## References
```txt
@article{raissi2019physics,
  title={Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations},
  author={Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E},
  journal={Journal of Computational physics},
  volume={378},
  pages={686--707},
  year={2019},
  publisher={Elsevier}
}
```


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

These resources are which I would like to give credit to.
* [Simple PyTorch Implementation of PINN](https://github.com/nanditadoloi/PINN)
* [Intro to PINNs](https://towardsdatascience.com/solving-differential-equations-with-neural-networks-afdcf7b8bcc4)
* [Physics Informed Deep Learning](https://maziarraissi.github.io/PINNs/)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->
## Contact
<div align="center">
  <h3>Please feel free to connect with me on social media</h3>
    <a href="https://github.com/ning-yu-kao">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-github.png" width="3%"/>
    </a>
    <img width="3%" />
    <a href="https://www.linkedin.com/in/kaoningyupage">
        <img src="https://github.com/ultralytics/yolov5/releases/download/v1.0/logo-social-linkedin.png" width="3%"/>
    </a>
    <img width="3%" />

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/ning-yu-kao/pinns-implements/blob/main/LICENSE
