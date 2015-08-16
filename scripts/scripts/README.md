# Scripts

So, here you will find all the scripts required to get the data prepared for deep learning.

However, it does not contain the .csv files required to run the scripts. If you require this, you will need to fight me in a battle to the death; for your reward will be the files.

Or just ask nicely, and I'll probably email it to you.

## Note(s)

* For some strange reason, the data handed over had commas within the filename which made the entire process a complete and utter nightmare. In the ``filePreperation`` directory, you will find a file called ```removethecommas.sh``` which can be run in the directory containing the images. It will remove all commas, and is required for use with all other scripts used here.
* I've made the entire process of copying files into folders with their label automated with the ```automated.sh``` bash script found in ```filePreperation```. Just run this in the directory with your image data in and the produced csv file you made with the python scripts in ```pythonFiles``` and it'll do it all for you.
* If you have any issues, email me or start a new issue and I'll fix it as soon as I can.
