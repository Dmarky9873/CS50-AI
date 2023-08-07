x = dict()

# Convert the list into a tuple
key = ([1, 2], 2)

# Add the tuple as a key to the dictionary
x[key] = 'some_value'

# Check if the tuple exists as a key in the dictionary
if key in x:
    print('yes')
