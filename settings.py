_RIGHT_TS_NAME = None
_USER_ID_NAME = None
_TARGET_NAME = None


def set_col_names(right_ts=None, user_id=None, target=None):
    global _RIGHT_TS_NAME
    global _USER_ID_NAME
    global _TARGET_NAME
    _RIGHT_TS_NAME = right_ts
    _USER_ID_NAME = user_id
    _TARGET_NAME = target


def get_right_ts_name(right_ts=None, none_permitted=False):
    if right_ts:
        return right_ts
    if _RIGHT_TS_NAME is None and not none_permitted:
        msg = "\Right time-stamp name has not been set yet."
        msg += "\nUse 'set_right_ts_name' to do so."
        msg += "\n\nEx: set_right_ts_name('fecha_der')"
        msg += "\n\nYou can also just pass it to the function as 'right_ts'."
        raise ValueError(msg)
    return _RIGHT_TS_NAME


def get_target_name(target=None, none_permitted=False):
    if target:
        return target
    if _TARGET_NAME is None and not none_permitted:
        msg = "\nTarget name has not been set yet."
        msg += "\nUse 'set_target_name' to do so."
        msg += "\n\nEx: set_target_name('rt_real_result')"
        msg += "\n\nYou can also just pass it to the function as 'target'."
        raise ValueError(msg)
    return _TARGET_NAME


def get_user_id_name(user_id=None, none_permitted=False):
    if user_id:
        return user_id
    if _USER_ID_NAME is None and not none_permitted:
        msg = "\nUser ID name has not been set yet."
        msg += "\nUse 'set_USER_ID_NAME' to do so."
        msg += "\n\nEx: set_USER_ID_NAME('costumer_id')"
        msg += "\n\nYou can also just pass it to the function as 'user_id'."
        raise ValueError(msg)
    return _USER_ID_NAME
