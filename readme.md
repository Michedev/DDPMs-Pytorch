```mermaid
  graph TD;
      T(Time step)-->B(Transformer Sinuosidal positional embedding);
      X-->R1(Res Block);
      B--> |Linear| R1;
      R1--> |Max Pooling 2x2| R2(Res Block);
```
