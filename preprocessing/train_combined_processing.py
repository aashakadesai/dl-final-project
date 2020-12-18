import numpy as np
import vggish_input
import csv
from pydub import AudioSegment

fnames = []
labels = []

#read in info from given csv file
#for verified dataset, add condition if row[2] == 1
with open('train_data_list.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	for row in csv_reader:
		if line_count == 0:
			line_count = line_count + 1
		else:
			fnames.append(row[0])
			labels.append(row[1])
			line_count = line_count + 1

print(len(fnames))

#write to file to examine data attributes
#we want monochannel, 44.1 Hz, 16-bit
'''
with open('train_data_list_att.csv', mode='w') as csv_file:
	csv_writer = csv.writer(csv_file, delimiter=',')
	print(len(fnames))
	for i in range(len(fnames)):

		f = 'audio_train/' + fnames[i]
		label = labels[i]
		sound = AudioSegment.from_wav(f)
		duration = sound.duration_seconds
		channel = sound.channels
		width= sound.sample_width
		fr = sound.frame_rate
		csv_writer.writerow([fnames[i], label, duration, channel, width, fr])
'''
count = 0
ms_names = []
ms_labels = []
ms_count = 0
mset_count = 0

#convert selected files to melspectograms
for i in range(len(fnames)):
	f = 'audio_train/' + fnames[i]
	label = labels[i]

	examples = vggish_input.wavfile_to_examples(f)
	if ms_count == 0:
		ms_data = examples
	else:
		#print("here")
		ms_data = np.concatenate((ms_data, examples))
	ms_count = ms_count + examples.shape[0]

	if ms_count > 500:
		name = 'train_combined/MSet' + str(mset_count) + '.npy'
		np.save(name, ms_data)
		print('SAVED')
		ms_count = 0
		mset_count = mset_count + 1

	for mel_spec in examples:
		name = 'MS' + str(count)
		ms_names.append(name)
		ms_labels.append(label)
		count = count + 1

name = 'train_combined/MSet' + str(mset_count) + '.npy'
np.save(name, ms_data)
print('SAVED')
ms_count = 0
mset_count = mset_count + 1

#write out melspec file names with label
with open('train_data_ms_list.csv', mode = 'w') as ms_file:
	ms_writer = csv.writer(ms_file, delimiter=',')
	for i in range(len(ms_names)):
		ms_writer.writerow([ms_names[i], ms_labels[i]])
