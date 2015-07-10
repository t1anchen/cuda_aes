# cuda_aes (WIP)

## A CUDA based AES Implementation
This prject comes from a fork of [cbguder](https://github.com/cbguder)'s [aes-on-cuda](https://github.com/cbguder/aes-on-cuda) project.

## License
This project is released under [The BSD 3-Clause License](http://www.opensource.org/licenses/BSD-3-Clause).

## Dependencies
Tested on Ubuntu 14.04 LTS
- gcc >= 4.8.2
- CUDA nvcc >= 5.5
- Python >= 2.7 

### About NVIDIA CUDA Installation on Ubuntu 10.10
1. Press `Ctrl + Alt + F1` to switch your workplace from gdm to plain txt tty
2. Type `sudo gdm stop` to make gdm stop
3. `sudo apt-get --purge remove nvidia*`
   Note: This step is very important! Because if not, you'll find Ubuntu 
         will recover the kernel module and Xorg configuration file once
	 you reboot.
4. If you have only one CUDA device, you should make gdm stop or kill X server.  The cuda-gdb should be able to debug on a CUDA device without working on X server.
