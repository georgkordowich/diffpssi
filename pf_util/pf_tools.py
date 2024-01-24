"""
File that contains utility functions for the PowerFactory API.
"""
import os
import sys

PF_PATH = r'C:\Program Files\DIgSILENT\PowerFactory 2022 SP3\Python\3.9'
sys.path.append(PF_PATH)

try:
    import powerfactory
except ImportError:
    print()
    raise ImportError('PowerFactory not found at: ', PF_PATH,
                      '\n Please change the PF_PATH variable in pf_tools.py to the correct location of '
                      'your PowerFactory installation.')

pf = powerfactory.GetApplication()
pf.Show()


def reset_project(project_path):
    """
    Function deletes the PowerFactory project and re-imports it, to avoid powerfactory using too much memory.
    """
    pf.ActivateProject(os.path.basename(project_path).split('.')[0] + '.IntPrj')

    # Get Active Project
    active_project = pf.GetActiveProject()

    # Delete the project completely in order to avoid a powerfactory memory leak.
    if active_project is not None:
        active_project.Deactivate()
        active_project.Delete()

    # Import the same project again, which is clean.
    location = pf.GetCurrentUser()
    import_obj = location.CreateObject('CompfdImport', 'Import')
    import_path = os.path.abspath(project_path)
    import_obj.SetAttribute("e:g_file", import_path)
    import_obj.g_target = location
    import_obj.Execute()
    import_obj.Delete()
    pf.ClearRecycleBin()
    pf.ActivateProject(os.path.basename(project_path).split('.')[0] + '.IntPrj')
    active_project = pf.GetActiveProject()

    if active_project is None:
        raise ImportError('Project could not be loaded. Please check the path to the project.')

    print('Purge successful!')

    active_sc = pf.GetActiveStudyCase()
    grid = active_project.GetContents('Nine-bus System.ElmNet', 1)[0]

    param_ident = active_project.GetContents('*.ComIdent', 1)[0]

    return active_project, active_sc, param_ident, grid
