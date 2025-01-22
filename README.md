# Plasmo3Net: A Convolutional Neural Network-Based Algorithm for Detecting Malaria Parasites in Thin Blood Smear Images
[Plasmo3Net Web App](https://plasmo3net.streamlit.app)
### Authors
[Afolabi Owoloye<sup>1,2,3</sup>](https://www.linkedin.com/in/afolabi-owoloye-a1b8a5b5/)
[Funmilayo Ligali<sup>1,3</sup>](https://www.linkedin.com/in/samuel-olawale-olubode-6191a81aa/)
[Ojochenemi Enejoh<sup>5</sup>](https://www.linkedin.com/in/adewale-ogunleye-09029684/)
[Oluwafemi Agosile<sup>1</sup>](https://www.linkedin.com/in/samuel-olawale-olubode-6191a81aa/)
[Adesola Musa<sup>3</sup>](https://www.linkedin.com/in/adewale-ogunleye-09029684/)
[Oluwagbemiga Aina<sup>3</sup>](https://www.linkedin.com/in/adewale-ogunleye-09029684/)
[Taiwo Idowu<sup>2</sup>](https://scholar.google.com/citations?hl=en&user=ViS6ndQAAAAJ)
[Kolapo Oyebola<sup>1,3</sup>](https://www.linkedin.com/in/kolapo-oyebola-phd-67493836/)


<h6 style='color: red;'><sup>1</sup>Centre for Genomic Research in Biomedicine (CeGRIB), College of Basic and Applied Sciences, Mountain Top University, Ibafo, Nigeria.</h6>
<h6 style='color: red;'><sup>2</sup>Parasitology and Bioinformatics Unit, Department of Zoology, Faculty of Science, University of Lagos, Lagos, Nigeria.</h6>
<h6 style='color: red;'><sup>3</sup>Nigerian Institute of Medical Research, Lagos, Nigeria.</h6>
<h6 style='color: red;'><sup>4</sup>Department of Biochemistry, Faculty of Basic Medical Science, University of Lagos, Lagos, Nigeria.</h6>
<h6 style='color: red;'><sup>5</sup>Genetics, Genomics and Bioinformatics Department, National Biotechnology Research and Development Agency, Abuja, Nigeria.</h6>

<sup>1</sup> Centre for Genomic Research in Biomedicine (CeGRIB), College of Basic and Applied Sciences, Mountain Top University, Ibafo, Nigeria.<br>
<sup>2</sup> Parasitology and Bioinformatics Unit, Department of Zoology, Faculty of Science, University of Lagos, Lagos, Nigeria.<br>
<sup>3</sup> Nigerian Institute of Medical Research, Lagos, Nigeria.<br>
<sup>4</sup> Department of Biochemistry, Faculty of Basic Medical Science, University of Lagos, Lagos, Nigeria.<br>
<sup>5</sup> Genetics, Genomics and Bioinformatics Department, National Biotechnology Research and Development Agency, Abuja, Nigeria.<br>

### Abstract
Early diagnosis of malaria is crucial for effective control and elimination efforts. Microscopy is a reliable field-adaptable malaria diagnostic method. However, microscopy results are only as good as the quality of slides and images obtained from thick and thin smears. In this study, we developed deep learning algorithms to identify malaria-infected red blood cells (RBCs) in thin blood smears. Three algorithms were developed based on a convolutional neural network (CNN). The CNN was trained on 15,060 images and evaluated using 4,000 images. After a series of fine-tuning and hyperparameter optimization experiments, we selected the top-performing algorithm, which was named Plasmo3Net. The Plasmo3Net architecture was made up of 13 layers: three convolutional, three max-pooling, one flatten, four dropouts, and two fully connected layers, to obtain an accuracy of 99.3%, precision of 99.1%, recall of 99.6%, and F1 score of 99.3%. The maximum training accuracy of 99.5% and validation accuracy of 97.7% were obtained during the learning phase. Four pre-trained deep learning models (InceptionV3, VGG16, ResNet50, and ALexNet) were selected and trained alongside our model as baseline techniques for comparison due to their performance in malaria parasite identification. The topmost transfer learning model was the ResNet50 with 97.9% accuracy, 97.6% precision, 98.3 % recall, and 97.9% F1 score. The accuracy of the Plasmo3Net in malaria parasite identification highlights its potential for automated malaria diagnosis in the future. With additional validation using more extensive and diverse datasets, Plasmo3Net could evolve into a diagnostic workflow suitable for field applications.
