import numpy as np
import vggish_input
import csv
from pydub import AudioSegment

audioPath = 'TUT-SED/audio/TUT-SED-synthetic-2016-mix-'
annotPath = 'TUT-SED/meta/TUT-SED-synthetic-2016-annot-'
trainStart = 0
trainStop = 20
count = 0 
ms_names = []
ms_labels = []


#given samples are 32 bit depth, vggish is for 16 bit depth
#so we resample the data
for i in range(trainStart, trainStop):
	fname = audioPath + str(i) + '.wav'
	sound = AudioSegment.from_wav(fname)
	resamp = sound.set_sample_width(2)
	resamp.export(audioPath + str(i) + 'resamp.wav', format='wav')


for i in range(trainStart, trainStop):
	#read in files and convert to spectograms
	fname = audioPath + str(i) + 'resamp.wav'
	examples = vggish_input.wavfile_to_examples(fname)

	specCount = examples.shape[0]
	
	labels = []

	for j in range(specCount):
		labels.append("")

	annotName = annotPath + str(i) + '.txt'

	#here we go through annotations file and assign labels to each spectograms
	with open(annotName, 'r') as annotFile:
		data = annotFile.readlines()

		for line in data:
			words = line.split()
			start = int((float(words[0]) * 1000)/960)
			stop = int((float(words[1]) * 1000)/960) + 1

			for k in range(start, stop):
				curr = labels[k]
				if words[2] not in curr:
					labels[k] = curr + ' ' + words[2]

	for melspec in examples:
		name = 'MS' + str(count)
		ms_names.append(name)
		count = count + 1

	ms_labels = ms_labels + labels

	#save spectograms as np array
	np.save('polyphonic/test/MSet' + str(i) + '.npy', examples)

#save spectogram names and labels in csv file
with open('test_data_poly_list.csv', mode = 'w') as ms_file:
	ms_writer = csv.writer(ms_file, delimiter=',')
	for i in range(len(ms_names)):
		ms_writer.writerow([ms_names[i], ms_labels[i]])








