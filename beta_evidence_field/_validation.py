def require(condition : bool,
            error     : type[Exception],
            message   : str) -> None:

    if not condition:
        raise error(message)


def require_all(requirements : list[tuple[bool, type[Exception], str]]) -> None:

    for condition, error, message in requirements:
        require(condition, error, message)