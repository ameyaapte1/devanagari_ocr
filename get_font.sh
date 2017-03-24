#!/bin/bash
fc-list :lang=hi | grep Devanagari_font | gawk -F ':' '{print $2" "$3}' | sort | sed 's/style=//' | sed 's/[a-zA-Z]*,//' | sed 's/^ /"/' | sed 's/$/" \\/' | sed 's/ [a-zA-Z]*,[a-zA-Z]*//'

