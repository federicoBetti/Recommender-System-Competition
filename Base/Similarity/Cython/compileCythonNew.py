import subprocess

from gevent import os

#compiledModuleSubfolder = "/Ba/Cython"
fileToCompile_list = ['Compute_Similarity_Cython.pyx']

for fileToCompile in fileToCompile_list:

    command = ['python',
               'compileCython.py',
               fileToCompile,
               'build_ext',
               '--inplace'
               ]

    output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd())

    try:

        command = ['cython',
                   fileToCompile,
                   '-a'
                   ]

        output = subprocess.check_output(' '.join(command), shell=True,
                                         cwd=os.getcwd())

    except:
        pass
