# Python IW-SSIM

Python implementation of the IW-SSIM image fidelity metric.

This implementation is algorithmically identical to the
[original](https://github.com/Jack-guo-xy/Python-IW-SSIM) by Xinyu Guo, with
some modern conveniences. The most notable convenience is the use of
[uv](https://astral.sh/uv) for dependencies, which allows easy venv management
as well as the ability to execute `iwssim.py` as a script.

## Usage

```sh
usage: iwssim.py [-h] [--cuda] [--double] [--no-iw] reference distorted
```

Example:

```sh
./iwssim.py images/Ref.bmp images/Dist.jpg
```

Which should print `IW-SSIM: 0.803887`.

## Credits

This implementation is under the [2-clause BSD license](LICENSE), same as the
original. For more information on IW-SSIM, see the
[Waterloo page](https://ece.uwaterloo.ca/~z70wang/research/iwssim) which links
to the paper.

The [original README](README.txt) is included in this repo for posterity.
