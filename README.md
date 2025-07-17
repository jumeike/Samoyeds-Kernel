# Samoyeds-Kernel: Dual-Side Sparse Matrix Multiplication Library for Samoyeds

Samoyeds-Kernel is the core high-performance library for dual-side sparse matrix multiplication, powering the [Samoyeds](https://github.com/guqiqi/Samoyeds) system. It is designed for large-scale sparse tensor computation scenarios, such as Mixture-of-Experts (MoE) models, and fully leverages GPU Sparse Tensor Cores.

**Paper**: [Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores](https://dl.acm.org/doi/10.1145/3689031.3717455)

## Install

### Pre-requisites
Samoyeds-Kernel requires the following dependencies:
- CUDA 11.4+
- CMake 3.18+
- GPUs with Sparse Tensor Core (such as NVIDIA Ampere or newer)

### Get the Code
```shell
git clone --recurse-submodules https://github.com/guqiqi/Samoyeds-Kernel.git Samoyeds-Kernel
cd Samoyeds-Kernel
```

### Build
To build Samoyeds-Kernel, run:
```shell
./build.sh
```

> If you encounter the following error:
> ```shell
> Policy "CMP0104" is not known to this version of CMake
> ```
> Please comment out the line `cmake_policy(SET CMP0104 OLD)` in `./benchmark/third_party/venom/include/sputnik/CMakeLists.txt`.

### Run
To run Samoyeds-Kernel, use:
```shell
./build/./benchmark/./horizontal_ssmm_benchmark -m 1024 -k 4096 -n 4096 --vector_length 128 --seed 42 -t
```
Example output:
```
Time to execute horizontal SSMM kernels 10000 times: 1.22182 s
```

## LICENCE

This project is licensed under the Apache License 2.0. See the [LICENCE](./LICENCE) file for details.

## Citation

If you use Samoyeds-Kernel in your research, please cite our paper:

```bibtex
@inproceedings{2025samoyeds,
  title={Samoyeds: Accelerating MoE Models with Structured Sparsity Leveraging Sparse Tensor Cores},
  author={Wu, Chenpeng and Gu, Qiqi and Shi, Heng and Yao, Jianguo and Guan, Haibing},
  booktitle={Proceedings of the Twentieth European Conference on Computer Systems},
  pages={293--310},
  year={2025}
}
```

## Contact

For questions or collaboration, please feel free to contact:
- [cpwu_sjtu@sjtu.edu.cn](mailto:cpwu_sjtu@sjtu.edu.cn)
- [qiqi.gu@sjtu.edu.cn](mailto:qiqi.gu@sjtu.edu.cn)
