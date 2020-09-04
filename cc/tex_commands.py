'''cc.tex.interpret() is capable of processing the commands in this module.'''

########################################################################

def newcommand( subsequent_tex ):
    '''Handle the LaTeX command \\newcommand{\\A}{B}. Will generate a
    new command \\A that will always result in output B.
    The original call to \\newcommand will be removed from the tex.

    Args:
        subsequent_tex (str):
            All LaTex following the \\newcommand call.

    Returns:
        command_tex (str):
            What to replace the call to \\newcommand with.

        dj (int):
            How much of subsequent_tex is the argument for \\newcommand
            I.e. subsequent_tex[:dj] is the input to newcommand.

        new_commands (dict of one function):
            The new command created by processing new command
    '''

    command_name = ''
    output = ''
    bracket_count = 0
    outer_bracket_count = 0
    args_tex = ''
    for i, c in enumerate( subsequent_tex ):
        
        # Keep track of brackets
        if c == '{' and subsequent_tex[i-1] != '\\':
            if bracket_count == 0:
                outer_bracket_count += 1
            bracket_count += 1

            # Outermost brackets are part of the command creation
            # don't process them
            if bracket_count == 1:
                continue

        elif c == '}' and subsequent_tex[i-1] != '\\':
            bracket_count -= 1
            if bracket_count == 0:
                outer_bracket_count += 1

            # Outermost brackets are part of the command creation
            # don't process them
            if bracket_count == 0:
                continue

        # Command name
        if outer_bracket_count == 1:
            
            # Don't include the \
            if command_name == '' and c == '\\':
                continue

            command_name += c

        # Arguments for new command
        elif outer_bracket_count == 2:
            args_tex += c

        # Macro itself
        elif outer_bracket_count == 3:
            output += c

        # New command should be done
        elif outer_bracket_count == 4:
            # The command should be ended
            assert not subsequent_tex[i+1].isalpha()
            break

    # Process the args_tex if given
    if args_tex != '':

        # Check for proper formatting
        assert args_tex[0] == '[' and args_tex[2] == ']'

        n_allowed_args = int( args_tex[1] )
    else:
        n_allowed_args = 0

    # Create the new command
    def command_fn( tex_in ):

        max_bracket_count = n_allowed_args * 2

        outer_bracket_count = 0
        bracket_count = 0
        args = []
        stack = ''
        for i, c in enumerate( tex_in ):

            # Keep track of brackets
            if c == '{' and subsequent_tex[i-1] != '\\':
                if bracket_count == 0:
                    outer_bracket_count += 1
                bracket_count += 1

                # Outermost brackets are part of the macro
                # don't process them
                if bracket_count == 1:
                    continue

            elif c == '}' and subsequent_tex[i-1] != '\\':
                bracket_count -= 1
                if bracket_count == 0:
                    outer_bracket_count += 1

                # Outermost brackets are part of the macro
                # don't process them
                if bracket_count == 0:
                    args.append( stack )
                    stack = ''
                    continue

            # When done
            if outer_bracket_count >= max_bracket_count:
                break

            # When retrieving args to pass
            if bracket_count > 0:
                stack += c

        # Count the length of the arguments
        dj = i

        # Modify the output
        used_output = ''
        skip_next = False
        for k, c in enumerate( output ):

            # This allows us to skip the actual number acting as an arg
            if skip_next:
                skip_next = False
                continue

            if c == '#' and output[k-1] != '\\':
                arg_ind = int(output[k+1]) - 1
                used_output +=  args[arg_ind]
                skip_next = True
            else:
                used_output += c

        return used_output, dj, {}

    # Format output
    command_tex = ''
    dj = i
    new_commands = { command_name: command_fn }

    return command_tex, dj, new_commands

########################################################################

