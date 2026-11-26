# Runtime evidence

The Python runner records the compiler's result for the source it receives.
A compilation error, an admitted proof, or a theorem dependency outside the
declared axiom policy prevents a theorem from being reported as verified.
The tests exercise acceptance of a valid declaration and rejection of
invalid, admitted, and unsupported declarations using the real compiler.

The runner retains failed compilation diagnostics. Discovered proof-shaped
outputs are inventoried by digest as unverified source. Algorithm test cases
compile and execute checks against expected values. Those comparisons verify
the tested cases. Universal statements require separately checked proofs.
