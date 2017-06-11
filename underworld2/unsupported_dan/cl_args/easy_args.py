
def easy_args(sysArgs, _dict):
    """
    Command line args are parsed and values go into the dictionary provided, under key names provided.
    Names are provided in the easyDict format, i.e. dictName.foo, rather than dictName['foo']
    This function will perform different updates, depending on the operator provided:

    * assign new value: =
    * in place multiply: *=

    Parameter
    ---------
    sysArgs : list
        the list of command line args provide by sys.argv (through the sys module)
    _dict:
        a dictionary to update values within

    Examples
    ---------

    >>>python script.py dp.arg3=3.0
    ...
    import sys
    from easydict import EasyDict as edict
    sysArgs = sys.argv
    dp = edict({})
    dp.arg3=2.0
    easy_args(sysArgs, [dp])

    ...The above code would replace the current value (2.0) of dictionary dp, key 'arg3', with the float 3.0

    python script.py dp.argFoo='Terry'
        This will replace the current value of dictionary dp, key 'argFoo', with the string 'Terry'

    python script.py dp.arg3*=3.0
        This will multiply the current value of dictionary dp, key 'arg3', with the float 3.0

    """


    #print(sysArgs)
    for farg in sysArgs:
        #Try to weed out some meaningless args.

        #print(farg)

        if ".py" in farg:
            continue
        if "=" not in farg:
            continue

        try:

            #########
            #Split out the dict name, key name, and value
            #########

            (dicitem,val) = farg.split("=") #Split on equals operator
            (dic,arg) = dicitem.split(".")
            if '*=' in farg:
                (dicitem,val) = farg.split("*=") #If in-place multiplication, split on '*='
                (dic,arg) = dicitem.split(".")

            #print(dic,arg,val)

            #########
            #Basic type conversion to float, boolean
            #########

            if val == 'True':
                val = True
            elif val == 'False':     #First check if args are boolean
                val = False
            else:
                try:
                    val = float(val) #next try to convert  to a float,
                except ValueError:
                    pass             #otherwise leave as string

            #########
            #Update the given dictionary
            #########

            try:
                if '*=' in farg:
                        _dict[arg] = _dict[arg]*val #multiply parameter by given factor
                else:
                        _dict[arg] = val    #or reassign parameter by given value
            except:
                    pass

        except:
            pass
