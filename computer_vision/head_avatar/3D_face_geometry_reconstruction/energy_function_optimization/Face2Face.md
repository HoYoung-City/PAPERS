# Face2Face

---
## Information
| Title |  Face2Face: Real-Time Face Capture and Reenactment of RGB Videos |
| -------------------------------------------- | ----------------------------------------------------- |
| `Conference or Journals`                                 | IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016                                          |
| `Date`                                  | 2016 |
| `Authors`                                   | Justus Thies, Michael Zollhöfer, Marc Stamminger, Christian Theobalt, Matthias Nießner |
| `Authors' affiliation`                                |                               |
| `Keywords`                                |  |
| `project website`                                    | https://niessnerlab.org/projects/thies2016face.html                                          |
| `code`                           |  |
| `paper` | https://niessnerlab.org/papers/2016/1facetoface/thies2016face.pdf|
---

## Task Overview
### 🎯 Goal
  - Real-time facial reenactment using only RGB video inputs.
### 📥 Input:
  - Target video
    - Pre-recorded monocular video (e.g, Youtube video)
  - Source video: 
    - Monocular video captured live with a commodity webcam.
### 📤  Output:
  - Animate the facial expressions of the target video by a source actor
  - Re-render the manipulated output video in a photo-realistic fashion

## 💡 Motivations
- Instead of transferring facial expressions to virtual CG characters, our main contribution is **monocular facial reenactment** in **real-time**.
- Online transfer of facial expressions of a source actor captured by an RGB sensor to a target actor.
- Faithful photo-realistic facial reenactment is the foundation for a variety of applications;
    - for instance, in video conferencing, the video feed can be adapted to match the face motion of a translator, or face videos can be convincingly dubbed to a foreign language.


## 🔧 How to Solve?
1. Optimize identity from video sequences (Offline / Non real-time)
2. facial expression tracking (Real-time)
3. Expression transfer for reenactment (Real-time)
4. re-render the synthesized target face (Real-time)
- 🛠 Detailed Breakdown:
    - Preprocessing, 
        - address the under-constrained problem of **facial identity recovery** from monocular
    video by **non-rigid model-based bundling**.
            - 📌 Bundling: The process of jointly optimizing camera poses, 3D structure, and object identity by using information from multiple video frames.
    - ⏱ Runtime, 
        - track facial expressions of both source and target video using a **dense photometric consistency measure**.
        - Reenactment is then achieved by fast and efficient **deformation transfer** between source and target.
        - **re-render** the synthesized target face on top of the corresponding video stream such that it seamlessly blends with the real-world illumination
    - 👄 Mouth interior
        - The mouth interior that best matches the re-targeted expression is retrieved from the target sequence and warped to produce an accurate fit.

## 🧪 Contributions
- dense, global non-rigid model-based bundling
- accurate tracking, appearance, and lighting estimation
in unconstrained live RGB video
- person-dependent expression transfer using subspace
deformations
- and a novel mouth synthesis approach.

---

## 📈 Method
### 1.Synthesis of facial imagery

#### Multi-linear Face Model (Used in Face2Face)

Face2Face uses a **multi-linear 3D face model** to represent and manipulate facial geometry, appearance, and expression.

#### Model Structure
$\mathrm{Face}(x) = \mathbf{a} + \mathbf{E}_{\mathrm{id}} \cdot \boldsymbol{\alpha} + \mathbf{E}_{\mathrm{alb}} \cdot \boldsymbol{\beta} + \mathbf{E}_{\mathrm{exp}} \cdot \boldsymbol{\delta}$


| 요소                                                      | 설명                             |
| ------------------------------------------------------- | ------------------------------ |
| $\mathbf{a}$                                            | 평균 얼굴 모양 (mean shape / albedo) |
| $\mathbf{E}_{\text{id}} \in \mathbb{R}^{3n \times 80}$  | 정체성(identity) 형태의 주성분(기저)      |
| $\boldsymbol{\alpha} \in \mathbb{R}^{80}$               | 개별 사람의 정체성을 나타내는 계수            |
| $\mathbf{E}_{\text{alb}} \in \mathbb{R}^{3n \times 80}$ | 피부 색상(albedo)의 기저              |
| $\boldsymbol{\beta} \in \mathbb{R}^{80}$                | 반사율(albedo) 계수                 |
| $\mathbf{E}_{\text{exp}} \in \mathbb{R}^{3n \times 76}$ | 표정(expression) 기저              |
| $\boldsymbol{\delta} \in \mathbb{R}^{76}$               | 표정 계수 (실시간으로 추적)               |
The face is parameterized as a combination of:


* **Identity (shape):**
  - $\mathbf{a}_{\mathrm{id}} + \mathbf{E}_{\mathrm{id}} \cdot \boldsymbol{\alpha}$
  - $\mathbf{E}_{\text{id}} \in \mathbb{R}^{3n \times 80}$, $\boldsymbol{\alpha} \in \mathbb{R}^{80}$

* **Albedo (skin color):**
  - $\mathbf{a}_{\text{alb}} + \mathbf{E}_{\mathrm{alb}} \cdot \boldsymbol{\beta}$
  - $\mathbf{E}_{\text{alb}} \in \mathbb{R}^{3n \times 80}$, $\boldsymbol{\beta} \in \mathbb{R}^{80}$

* **Expression:**
  - $\mathbf{E}_{\text{exp}} \cdot \boldsymbol{\delta}$

  - $\mathbf{E}_{\text{exp}} \in \mathbb{R}^{3n \times 76}$, $\boldsymbol{\delta} \in \mathbb{R}^{76}$

> ✅ Total mesh: **53K vertices**, **106K faces**

---

#### Rendering Pipeline

* **Rigid transformation:** $\Phi(v)$, camera projection: $\Pi(v)$
* **Illumination:** Approximated using **Spherical Harmonics (SH)** (1st 3 bands)
* **Output image:** Rasterization of 3D face model under lighting and camera params

---

#### Parameters Summary

```text
P = { α (identity), β (albedo), δ (expression), γ (lighting), R, t (pose), κ (camera) }
```

---

### 2. Energy Formulation

Face2Face reconstructs all unknown parameters $\mathbf{P}$ from monocular video via **robust variational optimization**.
The energy function is non-linear and composed of three terms:

---

#### Objective Function

$$
E(\mathbf{P}) = w_{\text{col}} \cdot E_{\text{col}} + w_{\text{lan}} \cdot E_{\text{lan}} + w_{\text{reg}} \cdot E_{\text{reg}}
$$

* $E_{\text{col}}$: Photometric consistency (pixel-wise appearance match)
* $E_{\text{lan}}$: Landmark alignment (feature points match)
* $E_{\text{reg}}$: Statistical regularization (parameter prior)

> **Weights:**
> $w_{\text{col}} = 1$, $w_{\text{lan}} = 10$, $w_{\text{reg}} = 2.5 \times 10^{-5}$

---

#### 🖼️ Photo-Consistency $E_{\text{col}}$

Measures pixel-level difference between input and synthesized images:

$$
E_{\text{col}} = \sum_{p \in V} \| C_S(p) - C_I(p) \|_2
$$

* $C_S$: Synthesized image
* $C_I$: Input RGB image
* $V$: Set of visible pixels
* Uses **$\ell_{2,1}$-norm** for robustness (L2 per pixel, L1 over image)

---

#### 🎯 Landmark Alignment $E_{\text{lan}}$

Aligns tracked 2D facial landmarks with projected 3D model points.

* Uses **facial landmark tracker** (Saragih et al.)
* Each 2D landmark $f_j$ maps to a 3D vertex $v_j = M_{\text{geo}}(\alpha, \delta)$
* Weighted by detection confidence $w_{\text{conf},j}$
* Helps guide optimization and avoid local minima

---

#### 📊 Statistical Regularization $E_{\text{reg}}$

Encourages parameters to stay near the mean of the multivariate normal prior:

- E_reg = || α / σ_id ||²₂ + || β / σ_alb ||²₂ + || δ / σ_exp ||²₂



* Prevents **implausible facial geometry or reflectance**
* Keeps optimization **stable and realistic**

---

### 3. Data-Parallel Optimization

Face2Face solves a **nonlinear, unconstrained optimization problem** for facial tracking in **real time** using a **data-parallel GPU-based solver**.

---

#### ⚙️ Optimization Method

* **Objective:** Robust nonlinear tracking energy
* **Solver:** IRLS (Iteratively Reweighted Least Squares)
* **Linearization:** Each iteration transforms into a nonlinear least-squares problem.

---

#### 🔁 Iterative Optimization Steps

1. **IRLS Iteration**

- ‖ r(P) ‖₂ = (‖ r(P_old) ‖₂)⁻¹ · ‖ r(P) ‖₁



* Fix residual weights
* Linearize objective
-  각 요소 설명
    * $r(\mathcal{P})$: 현재 파라미터 $\mathcal{P}$에서의 **잔차(residual)** 벡터
    * $r(\mathcal{P}_{\text{old}})$: 이전 반복(iteration)에서의 잔차 벡터
    * $\| \cdot \|_2$: L2 노름 (유클리디안 거리)
    * $\| \cdot \|_1$: L1 노름 (절댓값 합)
    * 괄호로 묶인 $\| r(\mathcal{P}_{\text{old}}) \|_2^{-1}$: 상수로 취급 → **가중치(weight)** 역할



2. **Gauss-Newton Step**

   * Solve:
$J^T J \delta^* = -J^T F$

   * Solver: **Preconditioned Conjugate Gradient (PCG)**
   * Output: Linear parameter update $\delta^*$

3. **Parameter Update**

   * Apply $\delta^*$ to update parameter vector $\mathbf{P}$

---

#### 🧩 Implementation Details

* **Jacobian $J$** and right-hand side $-J^T F$:
  Precomputed and stored in **GPU memory**
* **Framework:**

  * **DirectX 11** for rendering
  * **DirectCompute** for optimization
* **Optimization-friendly renderer:**

  * Uses **differential rasterization**
  * Stores **vertex & triangle attributes** to compute partial derivatives via **compute shaders**

---

### 4. Non-Rigid Model-Based Bundling

To robustly recover facial **identity** in a **monocular and under-constrained setting**, Face2Face applies a **non-rigid model-based bundling** strategy across multiple keyframes.

---

#### 🎯 Purpose

* Estimate **global identity** (shape & albedo) and **intrinsics**
* Separate identity from **pose, expression, illumination**

---

#### 📦 Estimated Parameters

For a set of $k$ keyframes, the following parameters are optimized:

| Type                  | Parameters                                                        |
| --------------------- | ----------------------------------------------------------------- |
| **Identity** (global) | $\alpha$, $\beta$ (shape, albedo)                                 |
| **Intrinsics**        | $\kappa$                                                          |
| **Per-frame**         | $\delta_k$, $R_k$, $t_k$, $\gamma_k$ (expression, pose, lighting) |

---

#### ⚙️ Optimization Strategy

* **Objective:** Joint energy across all keyframes
* **Solver:** PCG (Preconditioned Conjugate Gradient)
* **Jacobian:** Block-dense structure → efficiently exploited
* **Hierarchical Gauss-Newton:**

  * 3 levels of resolution:

    * Coarse (25 iterations)
    * Medium (5)
    * Fine (1)
  * 4 PCG steps per iteration

> 🔁 **Multilevel strategy** helps avoid local minima and improves convergence speed.

---

#### ⏱️ Performance

* Keyframes used: $k = 6$
* Total processing time: **\~20 seconds**
* **Linear scaling** with number of keyframes

---

#### ✅ Key Advantage

> Enables robust identity recovery **without prior camera calibration**, even under varying **illumination, pose, and expression**.

---

### 5. Expression Transfer

To map **expressions from source to target** while maintaining each actor’s personal facial traits, Face2Face uses a **subspace deformation transfer** technique.

---

#### 🎯 Goal

> Transfer expressions in **real time**
> Preserve **person-specific expression characteristics**

---

#### 📐 Method Overview

* Inspired by: **Sumner et al. (2004)** deformation transfer energy
* Works directly in the **expression blendshape subspace** (76D)
* Enables:

  * Fast computation
  * Precomputation of the **pseudo-inverse**
  * Low-dimensional optimization

---

#### 📊 Inputs and Outputs

| Role       | Data                                                           |
| ---------- | -------------------------------------------------------------- |
| **Input**  | Source identity $\alpha_S$, Target identity $\alpha_T$ (fixed) |
|            | Source expressions $\delta_S$: neutral + deformed              |
|            | Target neutral expression                                      |
| **Output** | Transferred expression $\delta_T$ in blendshape subspace       |

---

#### 🧮 Deformation Transfer Steps

1. **Compute deformation gradients**
   $A_i \in \mathbb{R}^{3 \times 3}$: per-triangle transformation for source (neutral → deformed)

2. **Formulate a least-squares problem**
   Find $\delta_T$ that minimizes error in triangle edge deformation

$$
\min_{\delta_T} \| A \delta_T - b \|^2
$$

   * $A \in \mathbb{R}^{6|F| \times 76}$: fixed matrix from mesh edges and blendshape basis
   * $b \in \mathbb{R}^{6|F|}$: computed each frame on GPU, based on $\delta_S$

3. **Solve using precomputed pseudo-inverse**

   * Use SVD to precompute:

$$
\delta_T = A^+ b
$$
   * Final solve: small **76 × 76** system — **real-time capable**

---

#### ✅ Advantages

* No need for extra smoothness terms (unlike Bouaziz et al.)
* Implicit smoothness via blendshape model
* Efficient GPU implementation
* **Plausible and person-specific expression output**


---

### 6. Mouth Retrieval

To generate a **realistic mouth region** for the reenacted face, Face2Face performs **image-based mouth retrieval and warping**.

---

#### 🎯 Objective

> Synthesize a **photo-realistic** target mouth that matches the transferred expression

---

#### 🧠 Method

1. **Retrieve** the best-matching mouth image from the **target video sequence**
2. **Warp** the retrieved mouth region to fit the reenacted facial pose

---

#### 📦 Assumptions

* The full target video or at least a **short segment** is available
* The target video contains **sufficient mouth shape variations**
* **Appearance of the target mouth is preserved**

  * ⚠️ **No source mouth copy**
  * ⚠️ **No generic 3D teeth model**

---

#### ✅ Advantages

* Maintains **person-specific realism**
* Avoids visual artifacts seen in:

  * **Mouth copy-paste** from source \[23]
  * **Generic 3D teeth proxies** \[8, 19]
* Produces **high-quality, natural-looking mouth synthesis**

---

> For full implementation details, refer to the original Face2Face paper (Section 10, Figure 4).


---
- **Since**: 2025.05.23  
- **Last updated**: 2025.05.23  
_(This .md file was written with the help of ChatGPT.)_
---
