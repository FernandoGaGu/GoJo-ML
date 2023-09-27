# Exceptions module
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
class IncorrectNumberOfClasses(Exception):
    """ Exception thrown when the input number of classes in an array is different from the predefined by the user. """
    def __init__(self, detected_classes: int, specified_classes: int, in_var: str = None):
        self.message = 'The number of detected classes ({}) does not match the number ' \
                       'of predefined classes ({}).'.format(detected_classes, specified_classes)

        if in_var is not None:
            self.message += ' Error in variable "{}"'.format(in_var)

        super().__init__(self.message)


class MissingArrayDimensions(Exception):
    """ Exception thrown when the input array does not match the expected number of dimensions """
    def __init__(self, expected_n_dims: int, input_n_dims: int, in_var: str = None):
        self.message = 'The number of dimensions ({}) does not match the expected number ' \
                       'of dimensions ({}).'.format(input_n_dims, expected_n_dims)

        if in_var is not None:
            self.message += ' Error in variable "{}"'.format(in_var)

        super().__init__(self.message)

