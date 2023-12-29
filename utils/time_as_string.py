import datetime


def get_current_time_as_string():
    return datetime.datetime.now().strftime("%B_%d_%Y_%I_%M%p")