'''cc.tex.interpret() is capable of processing the commands in this module.'''

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
    for i, c in enumerate( subsequent_tex ):
        
        # Process command
        if c == '{' or c == '}' and subsequent_tex[i-1] != '\\':
            bracket_count += 1
            continue

        # Command name
        if bracket_count == 1:
            
            # Don't include the \
            if command_name == '' and c == '\\':
                continue

            command_name += c

        elif bracket_count == 2:
            pass
        elif bracket_count == 3:
            output += c
        elif bracket_count == 4:
            # The command should be ended
            assert not subsequent_tex[i+1].isalpha()
            break

    # Create the new command
    def command_fn( tex_in ):
        '''Macros always return this output, regardless of what else follows.'''
        return output, 0, {}

    # Format output
    command_tex = ''
    dj = i
    new_commands = { command_name: command_fn }

    return command_tex, dj, new_commands
