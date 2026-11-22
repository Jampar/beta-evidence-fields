def require(condition : bool,
            error     : type[Exception],
            message   : str) -> None:

    if not condition: raise error(message)