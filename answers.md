# Answers

## Why would you use different color maps?

The pyplot library is not only for pictures but also for other graphs. There can be useful to show results in other color map. Also, we can show only red part of picture, then it will be also better use different colormap.

## How is inverting a grayscale value defined for uint8 ?

Don't understand the question. I think it is also 255 - ((r+g+b)/3)

## The histograms are usually normalized by dividing the result by the sum of all cells. Why is that?

Then the sum of all bins is 1 and the plot is probability distribution that random pixel belongs to the given bin

## Based on the results, which order of erosion and dilation operations produces opening and which closing?

If I close two elements, I create one big, so at first I have to joint with dilatation and then return back with erosion

If I open two elements, I separete theem, so at first I have to separete with erosion and them return back with dilation

## Why is the background included in the mask and not the object? How would you fix that in general? (just inverting the mask if necessary doesn’t count)

Because the object is darker than background. 

Ideas: 
+ check color of corners
+ choose minority color as white