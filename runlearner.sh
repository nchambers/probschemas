#!/bin/bash
#
#

if test -z "$2"; then
    echo "runlearner.sh -topics [num]"
    exit
fi

# Extra memory for java.
export MAVEN_OPTS="-Xmx3000m"

# Remove locks (only needed if doing expensive IR stuff)
if [ -e locks ]; then
    rm locks/*
fi

# Build the arguments to the java call
if (( $# > 0 )); then
    args="$@"
fi

# Make the call
echo mvn exec:java -Dexec.mainClass=nate.probschemas.Learner -Dexec.args="$args"
mvn exec:java -Dexec.mainClass=nate.probschemas.Learner -Dexec.args="$args"
