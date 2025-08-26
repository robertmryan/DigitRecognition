# Training small CNN for MNIST

1. Install Python 3.12.11.

    Note, I personally had Python 3.13 installed, but the `coremltools` package requires PyTorch 2.5 (not, for example, PyTorch 2.8) library. And the PyTorch 2.5 library requires Python 3.12. So, installed `pyenv` to manage multiple concurrent Python versions, and then installed Python 3.12:

    ```none
    $ python3 -V
    Python 3.13.7
    $ brew install pyenv
    …
    $ pyenv install 3.12.11
    …
    $ pyenv local 3.12.11
    $ python -V
    Python 3.12.11
    ```

2. I then installed PyTorch 2.5:

    ```none
    $ pip install --upgrade pip
    …
    $ pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cpu
    …
    ```

3. Then install `coremltools`:

    ```none
    $ pip install -U coremltools
    …
    ```

4. Train CNN:

    ```none
    $ python mnist_cnn_mps.py
    …
    ```

5. Create `.mlpackage`:

    ```none
    $ python create_ml_package_v2.py
    …
    ```

6. Copied that `MNISTSmallCNN_v2.mlpackage` into my Xcode project.

That results in a CoreML model that can be used in this project.
