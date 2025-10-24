# Python SVD Image Compression

This project demonstrates image compression using Singular Value Decomposition (SVD) in Python and was developed for a Matrix Computations course. It implements a "from scratch" SVD algorithm based on the **Power Iteration** method and compares its results (speed, compression ratio, image quality) against NumPy's highly optimized `linalg.svd` implementation.

The tool can compress both grayscale and color images from local files or URLs.

## Features

* **SVD Image Compression:** Reduces image file size by approximating the image matrix with fewer singular values.
* **Custom SVD Implementation:** Includes an educational implementation of SVD using Power Iteration and matrix deflation (`svd_compressor/svd.py`).
* **NumPy SVD Comparison:** Allows direct comparison with `numpy.linalg.svd`.
* **Color Image Support:** Compresses RGB images by applying SVD to each color channel independently.
* **URL or Local Image Support:** Can receive a path to a local or web-based image.
* **Performance Metrics:** Calculates and reports:
    * **Compression Time:** How long each SVD method takes.
    * **Compression Ratio:** Approximated based on the storage needed for truncated SVD components (U, s, Vh).
    * **PSNR (Peak Signal-to-Noise Ratio):** Measures the quality loss compared to the original image.
* **Modular Package:** Code is structured in a `svd_compressor` package.
* **CLI Interface:** Easy-to-use command-line interface via `argparse`.

## Project Structure

```
python-svd-image-compression/
├── .gitignore
├── LICENSE
├── README.md              # This documentation
├── requirements.txt       # Project dependencies
├── main.py                # Main runnable script (CLI)
└── svd_compressor/
    ├── __init__.py        # Makes 'svd_compressor' a package
    ├── svd.py             # Custom Power Iteration SVD implementation
    ├── compressor.py      # Image loading, SVD application, saving
    └── utils.py           # Helper functions, metrics (PSNR, ratio)
```

## How It Works

### 1. Singular Value Decomposition (SVD)

SVD is a fundamental matrix factorization technique. Any matrix $A$ (size $n \times m$) can be decomposed into three matrices:
```math
A = U \Sigma V^T
```
Where:
* $U$: An $n \times n$ orthogonal matrix (left singular vectors).
* $\Sigma$: An $n \times m$ diagonal matrix containing the **singular values** ($\sigma_1 \ge \sigma_2 \ge \dots \ge 0$) along its main diagonal.
* $V^T$: The transpose of an $m \times m$ orthogonal matrix $V$ (right singular vectors).

The singular values ($\sigma_i$) represent the "importance" or "energy" captured by their corresponding singular vectors. Larger singular values correspond to more significant features in the original matrix.

### 2. Image Compression with SVD

An image can be represented as a matrix (or multiple matrices for color channels) where each element is a pixel intensity. SVD allows us to approximate this matrix using fewer components, achieving compression.

1.  **Decomposition:** Apply SVD to the image matrix $A$.
2.  **Truncation (Low-Rank Approximation):** Keep only the largest $k$ singular values in $\Sigma$ and the corresponding first $k$ columns of $U$ and first $k$ rows of $V^T$. Let these be $U_k$, $\Sigma_k$, and $V^T_k$.
3.  **Reconstruction:** Reconstruct an approximate image matrix $A_k$:
    ```math
    A_k = U_k \Sigma_k V^T_k
    ```
    This $A_k$ is the closest rank-$k$ approximation of the original image $A$.
4.  **Storage:** Instead of storing the original $n \times m$ pixels, we store the much smaller matrices $U_k$ ($n \times k$), the $k$ singular values (vector $s_k$), and $V^T_k$ ($k \times m$). The total number of values stored is $(n \times k) + k + (k \times m)$, which is significantly less than $n \times m$ when $k$ is small.

The value of $k$ controls the trade-off:
* **Small $k$:** Higher compression, lower image quality.
* **Large $k$:** Lower compression, higher image quality.

### 3. Custom SVD: Power Iteration Method

The `svd_compressor/svd.py` file implements SVD using the Power Iteration method. This is **not** the most stable or efficient way but demonstrates a core concept.

1.  **Find Dominant Vector:** It uses power iteration on $A^T A$ (if $n > m$) or $A A^T$ (if $n \le m$) to find the eigenvector corresponding to the *largest* eigenvalue. This eigenvector is the first right (or left) singular vector ($v_1$ or $u_1$).
2.  **Calculate $\sigma_1$ and the other vector:** Once $v_1$ is found, calculate $u_1 = A v_1 / \sigma_1$, where $\sigma_1 = \| A v_1 \|$. (Or vice-versa if $u_1$ was found first).
3.  **Matrix Deflation:** "Remove" the component captured by the first singular value/vectors from the matrix: $A_{new} = A - \sigma_1 u_1 v_1^T$.
4.  **Repeat:** Apply power iteration to $A_{new}$ to find the *next* largest singular value/vectors ($\sigma_2, u_2, v_2$).
5.  Continue this process $k$ times.

**Limitations:** This method can suffer from numerical instability (errors accumulating during deflation) and is much slower than optimized algorithms (like those used in NumPy, which are based on QR decomposition or similar).

### 4. Comparison: Custom vs. NumPy

This tool allows direct comparison:
* **NumPy (`numpy.linalg.svd`):** Highly optimized, numerically stable, and fast. Uses sophisticated algorithms implemented in underlying libraries like LAPACK (often written in Fortran/C). This is the **standard** way to perform SVD.
* **Custom (`power_iteration_svd`):** Educational. Demonstrates the Power Iteration concept but is **slower** and can be **less accurate**, especially for finding smaller singular values due to deflation errors.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/msmrexe/python-svd-image-compression.git
    cd python-svd-image-compression
    ```

2.  **Create a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the compression:**
    Provide the path or URL to your image and the desired rank `k`.

    ```bash
    # Compress using both methods with k=50 (default output dir 'compressed_images/')
    python main.py path/to/your/image.jpg -k 50

    # Compress using only NumPy with k=20
    python main.py https://example.com/image.png -k 20 --method numpy

    # Compress using only the custom method with k=100 and specify output dir
    python main.py path/to/your/image.bmp -k 100 --method custom --output-dir ./output
    ```

    The script will print the time taken, compression ratio, and PSNR for each method run, and save the compressed image(s) to the output directory. Compare the output images and the metrics!
    
---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
