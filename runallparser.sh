#!/bin/bash
#


# Extra memory for java.
export MAVEN_OPTS="-Xmx4000m"


# Build the arguments to the java call
args=""
if (( $# > 0 )); then
    args="$args $@"
fi

# Make the call
mvn exec:java -Dexec.mainClass=nate.AllParser -Dexec.args="$args"
