import sys, os
from os import listdir
from os.path import isfile, join
import codecs


results_path = ''
out_path = ''


arguments_path = 'arguments/'
linking_path = 'linking/'


filenames = [f for f in listdir(results_path + arguments_path) if isfile(join(results_path + arguments_path, f))]

assert len(filenames) == 30002



def get_linking_arg_list(filename):
    linking_arg_list = set()
    with codecs.open(filename, 'rb', encoding='utf-8') as f:
        lines = f.read()
    lines = lines.split('\n')    
    for line in lines:
        if line == "":
            continue
        event, args = line.split('\t')
        #args_to_write = []
        args = args.split(' ')
        for arg in args:
            if arg in linking_arg_list:
                print 'uhoh, dup arg in ', filename
            else:
                linking_arg_list.add(arg)
    return linking_arg_list


missing_count = 0
has_args_count = 0

for filename in filenames:
    found_missing = False
    found_args = False
    arg_file = results_path + arguments_path + filename
    linking_file = results_path + linking_path + filename
    linking_arg_set = get_linking_arg_list(linking_file)
    with codecs.open(arg_file, 'rb', encoding='utf-8') as f:
        arg_lines = f.read()
    arg_lines = arg_lines.split('\n')
    for line in arg_lines:
        if line == "":
            continue
        found_args = True
        line_id, event_filename, event, role, cas, cas_offset, predicate_justification_offset, base_filler_offset, additional_arg_justification, realis, confidence = line.split('\t')
        if line_id not in linking_arg_set and realis != 'Generic':
            found_missing = True
    if found_missing:
        missing_count += 1
    if found_args:
        has_args_count += 1

print missing_count
print has_args_count













def get_linking_lines(filename, args_to_remove):
    with codecs.open(filename, 'rb', encoding='utf-8') as f:
        lines = f.read()
    lines = lines.split('\n')
    linking_lines_to_write = []
    for line in lines:
        if line == "":
            continue
        event, args = line.split('\t')
        args_to_write = []
        args = args.split(' ')
        for arg in args:
            if arg not in args_to_remove:
                args_to_write.append(arg)
            else:
                print 'deleting', arg, 'from event linking'
        if len(args_to_write) == 0:
            continue
        arg_string = ""
        for i, arg in enumerate(args_to_write):
            if i == (len(args_to_write) - 1):
                arg_string += arg
            else:
                arg_string += arg + ' '
        linking_lines_to_write.append(event + '\t' + arg_string + '\n')
    return linking_lines_to_write


# count lines in arg files:
total_event_args = 0
for filename in filenames:
    arg_file = results_path + arguments_path + filename
    linking_file = results_path + linking_path + filename
    argument_lines_to_write = []
    args_to_remove = []
    with codecs.open(arg_file, 'rb', encoding='utf-8') as f:
        arg_lines = f.read()
    arg_lines = arg_lines.split('\n')
    for line in arg_lines:
        if line == '':
            continue
        total_event_args += 1
        line_id, event_filename, event, role, cas, cas_offset, predicate_justification_offset, base_filler_offset, additional_arg_justification, realis, confidence = line.split('\t')

print 'total_event_args', total_event_args

# count args in linking files:
total_linking_args = 0
for filename in filenames:
    arg_file = results_path + arguments_path + filename
    linking_file = results_path + linking_path + filename
    #argument_lines_to_write = []
    #args_to_remove = []
    with codecs.open(linking_file, 'rb', encoding='utf-8') as f:
        lines = f.read()
        lines = lines.split('\n')
        for line in lines:
            if line == "":
                continue
            event, args = line.split('\t')
            #args_to_write = []
            args = args.split(' ')
            total_linking_args += len(args)

print 'total linking args ', total_linking_args
print 'difference is', str(total_event_args - total_linking_args)


if not os.path.exists(out_path + 'arguments'):
    os.makedirs(out_path + 'arguments/')

if not os.path.exists(out_path + 'linking'):
    os.makedirs(out_path + 'linking/')

if not os.path.exists(out_path + 'corpusLinking'):
    os.makedirs(out_path + 'corpusLinking/') 

with open(out_path + 'corpusLinking/corpusLinking', 'wb'):
    print 'Created empty corpusLinking file'

for filename in filenames:
    arg_file = results_path + arguments_path + filename
    linking_file = results_path + linking_path + filename
    arg_file_out = out_path + arguments_path + filename
    linking_file_out = out_path + linking_path + filename
    argument_lines_to_write = []
    args_to_remove = []
    with codecs.open(arg_file, 'rb', encoding='utf-8') as f:
        arg_lines = f.read()
    arg_lines = arg_lines.split('\n')
    for line in arg_lines:
        if line == "":
            continue
        line_id, event_filename, event, role, cas, cas_offset, predicate_justification_offset, base_filler_offset, additional_arg_justification, realis, confidence = line.split('\t')
        if event != 'Movement.Transport-Person' or role != 'Artifact':
            argument_lines_to_write.append(line + '\n')
        else:
            print 'removing argument:', event, role, line_id, event_filename, cas, filename
            args_to_remove.append(line_id)
    linking_lines_to_write = get_linking_lines(linking_file, args_to_remove)
    with codecs.open(arg_file_out, 'wb', encoding='utf-8') as f:
        for item in argument_lines_to_write:
            f.write(item)
    with codecs.open(linking_file_out, 'wb', encoding='utf-8') as f:
        for item in linking_lines_to_write:
            f.write(item)            


total_event_args_after = 0
for filename in filenames:
    arg_file = out_path + arguments_path + filename
    linking_file = out_path + linking_path + filename
    #argument_lines_to_write = []
    #args_to_remove = []
    with codecs.open(arg_file, 'rb', encoding='utf-8') as f:
        arg_lines = f.read()
    arg_lines = arg_lines.split('\n')
    for line in arg_lines:
        if line == '':
            continue
        total_event_args_after += 1
        line_id, event_filename, event, role, cas, cas_offset, predicate_justification_offset, base_filler_offset, additional_arg_justification, realis, confidence = line.split('\t')

print 'total_event_args', total_event_args_after


total_linking_args_after = 0
for filename in filenames:
    arg_file = out_path + arguments_path + filename
    linking_file = out_path + linking_path + filename
    #argument_lines_to_write = []
    #args_to_remove = []
    with codecs.open(linking_file, 'rb', encoding='utf-8') as f:
        lines = f.read()
        lines = lines.split('\n')
        for line in lines:
            if line == "":
                continue
            event, args = line.split('\t')
            #args_to_write = []
            args = args.split(' ')
            total_linking_args_after += len(args)

print 'total linking args ', total_linking_args_after
print 'difference is', str(total_event_args_after - total_linking_args_after)
