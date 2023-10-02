# Module with various functionalities used by the library.
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
# STATUS: uncompleted and not functional, still in development
#
from sklearn.model_selection import (
    RepeatedKFold,
    RepeatedStratifiedKFold,
    LeaveOneOut)


from .validation import checkMultiInputTypes


def getCrossValObj(cv: int = None, repeats: int = 1, stratified: bool = False, loocv: bool = False,
                   random_state: int = None) -> RepeatedKFold or RepeatedStratifiedKFold or LeaveOneOut:
    """ Function used to obtain the sklearn class used to perform an evaluation of the models according
    to the cross-validation or leave-one-out cross-validation (LOOCV) schemes.

    Parameters
    ----------
    cv : int, default=None
        (cross-validation) This parameter is used to specify the number of folds. Ignored when loocv
        is set to True.

    repeats : int, default=1
        (cross-validation) This parameter is used to specify the number of repetitions of an N-repeats
        cross-validation. Ignored when loocv is set to True.

    stratified : bool, default=False
        (cross-validation) This parameter is specified whether to perform the cross-validation with class
        stratification. Ignored when loocv is set to True.

    loocv : bool, default=False
        (Leave-one-out cross validation) Indicates whether to perform a LOOCV. If this parameter is set to
        True the rest of the parameters will be ignored.

    random_state : int, default=None
        (cross-validation) Random state for study replication.

    Returns
    -------
    cv_obj : RepeatedKFold or RepeatedStratifiedKFold or LeaveOneOut
        Cross-validation instance from the 'sklearn' library.
    """

    checkMultiInputTypes(
        ('cv', cv, [int, type(None)]),
        ('repeats', repeats, [int, type(None)]),
        ('stratified', stratified, [bool]),
        ('loocv', loocv, [bool]),
        ('random_state', random_state, [int, type(None)]))

    if loocv:
        return LeaveOneOut()
    else:
        if cv is None:
            raise TypeError(
                'Parameter "cv" in "gojo.util.tools.getCrossValObj()" must be selected to a integer if '
                'loocv is set to False.')

        if stratified:
            return RepeatedStratifiedKFold(n_repeats=repeats, n_splits=cv, random_state=random_state)
        else:
            return RepeatedKFold(n_repeats=repeats, n_splits=cv, random_state=random_state)



