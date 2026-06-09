# Necessary imports
from hessian.hessian import hessian
from itertools import islice

# Load data batches for Hessian computation, used in compute_top_and_bottom_hessian_eigenpairs.
def get_hessian_batch_tensor(loader, device, num_batches=8):
    xs, ys = [], []
    for x, y in islice(loader, num_batches):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, dim=0).to(device), torch.cat(ys, dim=0).to(device)


# Will give you top and bottom eigenvector
def compute_top_and_bottom_hessian_eigenpairs(
    model,
    loader,
    criterion,
    device,
    num_batches=8,
    tol=1e-3,
    maxiter=100,
    ncv=None,
):
    """
    Compute the largest algebraic Hessian eigenvalue and the smallest algebraic
    Hessian eigenvalue, together with their eigenvectors.

    This does NOT use absolute value ordering.

    Returns
    -------
    dict with:
        top_eigenvalue:
            Largest algebraic eigenvalue of H.
        top_eigenvector:
            List of tensors matching model.parameters().
        bottom_eigenvalue:
            Smallest algebraic eigenvalue of H. This is the most negative one
            if negative curvature exists.
        bottom_eigenvector:
            List of tensors matching model.parameters().
    """
    import numpy as np
    from scipy.sparse.linalg import LinearOperator, eigsh

    model.eval()

    inputs, targets = get_hessian_batch_tensor(
        loader,
        device,
        num_batches=num_batches,
    )

    model.zero_grad(set_to_none=True)

    hessian_comp = hessian(
        model,
        criterion,
        data=(inputs, targets),
        cuda=(device == "cuda"),
    )

    params = [p for p in model.parameters() if p.requires_grad]
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]
    total_dim = sum(numels)

    def _numpy_to_tensor_list(v_np):
        """Convert a flat numpy vector into a list of parameter-shaped tensors."""
        v_torch = torch.from_numpy(v_np).to(
            device=device,
            dtype=params[0].dtype,
        )

        vectors = []
        offset = 0
        for shape, numel in zip(shapes, numels):
            vectors.append(v_torch[offset:offset + numel].view(shape))
            offset += numel

        return vectors

    def _tensor_list_to_numpy(v_list):
        """Flatten a list of tensors into a CPU numpy vector."""
        return torch.cat([
            v.detach().reshape(-1).cpu()
            for v in v_list
        ]).numpy()

    def _matvec(v_np):
        """
        Matrix-vector product v -> H v.

        scipy eigsh calls this repeatedly. PyHessian supplies the HVP.
        """
        v_list = _numpy_to_tensor_list(v_np)

        model.zero_grad(set_to_none=True)

        hvp_result = hessian_comp.dataloader_hv_product(v_list)

        # PyHessian commonly returns: eigenvalue, Hv
        if isinstance(hvp_result, tuple):
            _, hv_list = hvp_result
        else:
            hv_list = hvp_result

        model.zero_grad(set_to_none=True)

        return _tensor_list_to_numpy(hv_list)

    H = LinearOperator(
        shape=(total_dim, total_dim),
        matvec=_matvec,
        dtype=np.float64,
    )

    # Largest algebraic eigenvalue, not largest by magnitude.
    top_vals, top_vecs = eigsh(
        H,
        k=1,
        which="LA",
        tol=tol,
        maxiter=maxiter,
        ncv=ncv,
    )

    # Smallest algebraic eigenvalue, not smallest magnitude.
    # This is the most negative eigenvalue if the Hessian has negative curvature.
    bottom_vals, bottom_vecs = eigsh(
        H,
        k=1,
        which="SA",
        tol=tol,
        maxiter=maxiter,
        ncv=ncv,
    )

    top_eigenvalue = float(top_vals[0])
    bottom_eigenvalue = float(bottom_vals[0])

    top_eigenvector = _numpy_to_tensor_list(top_vecs[:, 0])
    bottom_eigenvector = _numpy_to_tensor_list(bottom_vecs[:, 0])

    return {
        "top_eigenvalue": top_eigenvalue,
        "top_eigenvector": [v.detach().clone() for v in top_eigenvector],
        "bottom_eigenvalue": bottom_eigenvalue,
        "bottom_eigenvector": [v.detach().clone() for v in bottom_eigenvector],
    }