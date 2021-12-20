export fileid=1kgwhoReVF-eC_tuBhnDoFBpaNmFM_VGE
wget -O supervised.ipynb 'https://docs.google.com/uc?export=download&id='$fileid
ipython nbconvert supervised.ipynb --to rst --TagRemovePreprocessor.remove_all_outputs_tags='{"remove_output"}'
rm *.ipynb
mv supervised* experiments/


export fileid=1KKyCgDGujCYDJqLHcpCxezFwc4k165fl
wget -O sequential.ipynb 'https://docs.google.com/uc?export=download&id='$fileid
ipython nbconvert sequential.ipynb --to rst --TagRemovePreprocessor.remove_all_outputs_tags='{"remove_output"}'
rm *.ipynb
mv sequential* experiments/
