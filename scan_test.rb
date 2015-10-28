# Usage:
#     scan_test [TEST SIZE] [BLOCK WIDTH]
# Gives the expected result of scan

input = (1..(ARGV[0].to_i))
sum = 0;
output = input.map{|e| oldsum = sum; sum += e; oldsum}

print "h_in = [ "
for elem in input
    print "#{elem} "
end
puts "];"

print "h_out = [ "
for elem in output
    print "#{elem} "
end
puts "];"