
# EE-666: Assignment 3 - Panaroma Stitching

## Given a set of images, Find Homography matrices and stitch images to make a Panaroma.
This assignment is intended to provide practical experience with GitHub, along with the concepts discussed in class.


### Pre-requisites
 - Github Account.  Don't have one still? Create one. 
 - Install git.
 - Install Python locally in your system. Recommended : open-source distribution of the Python by Anaconda.
 - Install opencv ```pip install opencv-python```
 - Fork the repository @ `https://github.com/shash29-dev/ES666-Assignment3.git`  

```
# cd /path/to/folder/where/to/clone
git clone https://github.com/<username>/<forked-repo>.git 
```

## Check if boilerplate code works

```
cd <forked-repo>
python main.py
```

Running `main.py` should create `./results` folder and exit without Error. 


## Inside the repo
 - `Images : ` This folder contains images to be stitched to create panaroma.
 - `src` : Your Code goes here, Inside a folder. Check a Dummy Submissions by `JohnDoe`.
    - `JohnDoe/stitcher.py :` contains class `PanaromaStitcher`. Go through the class method named `make_panaroma_for_images_in` which should return two outputs: Final stitched Image and a List of Homography matrices.

    Note:  You can organise your code however you want but yout folder must have `stitcher.py` file containing class `PanaromaStitcher` with atleast one method named `make_panaroma_for_images_in` returning Final stitched Image and Homography matrices.

    - `main.py :` Main file to run all Implementations inside `src`.
    - Check other submssions by `JaneDoe` for stitching obtained by cv2 and `DarthVader` for a failure case.


## Create Your Stitcher

 - Check output related to`DarthVader's` submission. Like `DarthVader`, the `try` block in `main.py` should not fail. The returned outputs from `stitcher.py` should be in required order: stitched_image and a list of matrices. The `stitched_image` will get saved in `./results` folder.
 - You can delete dummy submssions included in Repo. 
 - Create a folder inside src with `stitcher.py` and complete the class method `make_panaroma_for_images_in` as discussed above.
 - Check `./results` for generated results.

Note: For Homography matrix estimation, use of library functions are not allowed. You can detect and match feature descriptors using utilities provided by existing libraries.


## Submission
 - Finally, Upload the code to the your repository. 

```
git add .
git commit -m "Final Submission"
git push origin main
```

## Evaluation
The final evaluation will be made on the content of `stitcher.py` and output generated by it. 
Please update the Colab file by replacing the line that performs the repository cloning with your forked repo, generate the results, and share them with us.


## Merge your fork with original repo (Optional) 
In case you want to merge your implementation of Panaroma stitching to https://github.com/shash29-dev/ES666-Assignment3.git, create a pull request. 
This must be requested after submission deadline. 

 - Learn about create pull request features of github. If there are no conflicts, The code will be merged. (In this case you migh want to push your code on a branch instead of main)
  ```
  git checkout -b <branch-name>
  git add .
  git commit -m "Final Submission"
  git push origin <branch-name>
  ```
  - Share your folder with me via mail. (Not recommended)


Why should you this? After your code is merged, `main.py` file can generate results for stitching of your implementation and others who opt to submit; along with `JanDoe`, `JohnDoe` and hopefully `DarthVader` if he is able to fix his error till submission deadline: 27th Oct 2024.