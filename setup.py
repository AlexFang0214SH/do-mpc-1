from setuptools import setup
from typing import List

# Read the do-mpc version
exec(open('do_mpc/_version.py').read())

# Utility to read the requirement files
def read_file_lines(file_name: str, task = 'Reading requirement files') -> List[str]:
    """ Read a requirements file and return the requirement specifiers it lists.

    Blank lines, ``#`` comments and ``-r``/``-c`` include directives are skipped,
    since setuptools would otherwise receive them as (invalid) requirement
    specifiers. Filtering here means requirements files can carry comments and
    can include each other without the caller having to slice by position.
    """
    try:
        with open(file_name, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file.readlines()]
        return [line for line in lines
                if line and not line.startswith('#') and not line.startswith('-')]
    except FileNotFoundError:
        print(f"Task {task} failed. File {file_name} not found.")
        return []
    except Exception as e:
        print(f"Task {task} failed. An error occurred: {str(e)}")
        return []

setup(
    name='do_mpc',
    version=__version__,
    packages=['do_mpc','do_mpc.controller','do_mpc.differentiator',
              'do_mpc.estimator','do_mpc.model','do_mpc.sampling',
              'do_mpc.sysid','do_mpc.tools', 'do_mpc.opcua',
             'do_mpc.approximateMPC'],
    author='Sergio Lucia and Felix Fiedler',
    author_email='sergio.lucia@tu-dortmund.de',
    url='https://www.do-mpc.com',
    license='GNU Lesser General Public License version 3',
    # README.md contains non-ASCII (e.g. 'Lueken' with an umlaut), so the encoding
    # must be stated explicitly: on a non-UTF-8 default locale this would
    # otherwise mojibake the PyPI description or raise UnicodeDecodeError.
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    install_requires= read_file_lines('requirements.txt'),
    extras_require = {
        # read_file_lines already drops the '-r requirements.txt' include, so no
        # positional slicing is needed here.
        'full': read_file_lines('requirements_full.txt'),
    }
)
