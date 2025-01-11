from src.Dim.PCA import spectral_pca_reduction
from src.Dim.kernel_PCA import spectral_kernel_pca_reduction
from src.Dim.MDS import spectral_mds_reduction
from src.Dim.UMAP import spectral_umap_reduction
from src.Dim.NMF import spectral_nmf_reduction


def apply_dimension_reduction(data, config, logger=None):
    """
    应用降维方法到数据

    Args:
        data: 输入数据
        config: 配置对象

    Returns:
        reduced_data: 降维后的数据
    """

    if not config.perform_dim_reduction:
        return data
    if logger is not None:
        logger.info(f"原始数据形状: {data.shape}")
    else:
        print(data.shape)

    if config.dim_reduction == 'PCA':
        reduced_data, _ = spectral_pca_reduction(data, n_components=config.n_components)
    elif config.dim_reduction == 'KernelPCA':
        reduced_data, _ = spectral_kernel_pca_reduction(data=data,
                                                        n_components=config.n_components,
                                                        kernel=config.kpca_kernel,
                                                        gamma=config.kpca_gamma,
                                                        random_state=config.seed)
    elif config.dim_reduction == 'MDS':
        reduced_data, _ = spectral_mds_reduction(data, n_components=config.n_components, random_state=config.seed)
    elif config.dim_reduction == 'UMAP':
        reduced_data = spectral_umap_reduction(data,
                                               n_components=config.n_components,
                                               n_neighbors=config.umap_n_neighbors,
                                               min_dist=config.umap_min_dist,
                                               random_seed=config.seed)
    elif config.dim_reduction == 'NMF':
        reduced_data, _ = spectral_nmf_reduction(data, n_components=config.n_components)
    else:
        raise ValueError(f"Unsupported dimension reduction method: {config.dim_reduction}")
    if logger is not None:
        logger.info(f"降维后的数据形状: {reduced_data.shape}")
    else:
        print(f"降维后的数据形状: {reduced_data.shape}")
    return reduced_data
