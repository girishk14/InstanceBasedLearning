import json

def make_control_file_Arry():
	cf = open('control.json' , 'w')	
	metadata = {}
	metadata['location'] = ['data/Voting/house-votes-84.data']
	metadata['class_name'] = 'Class'
	metadata['attr_names'] = ['N/A'] * 16
	metadata['attr_types'] = ['d'] * 16
	metadata['class_position'] = 0
	cf.write(json.dumps(metadata, indent=1))

make_control_file_Arry()
