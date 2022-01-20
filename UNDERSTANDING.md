# Here is what I have learned ?
**Inter-Attention** 
- It is basically same as LSTMs or RNNs but each output state s[i] is formed by a function of all input hidden states. 
- Something like : s[i] = sum_over_j(w[j] * h[j]. Now by back-prop we learn these params w.  

**Self-Attention**
- Now the problem of vanishing gradient is solved by since the path-length is still O(n), n is the seq_len.
- Inter Attention can't be parallelised.
- Next idea is to have self attention, what it means that `every word` in input sequence attends to some `other word` in the input sequence itself. 

**Blogs/Resources**
1. http://peterbloem.nl/blog/transformers
2. https://github.com/pbloem/former
3. https://www.youtube.com/watch?v=U0s0f995w14
4. https://www.youtube.com/watch?v=OyFJWRnt_AY

