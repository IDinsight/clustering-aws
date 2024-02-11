def create_ids(df_length: int, prefix: str = "GRID_") -> list[str]:
    """
    Create a list of string IDs in the format "{prefix}001".

    Parameters
    ----------
    df : dataframe containing grids

    Returns
    -------
    list : list of string_id_list.
    """

    # make list of ids [1, 2, 3, ...]
    ids = list(range(1, df_length + 1))

    # make list of ids with leading zeros ["001", "002", "003", ...]
    max_digits = len(str(df_length))
    string_ids = [prefix + str(id).zfill(max_digits) for id in ids]

    return string_ids
