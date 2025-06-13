---
title: 土豆提权
date: '2025-05-30 17:25:06'
permalink: /post/potato-rights-znsrmo.html
tags:
  - 土豆家族
  - 提权
  - windows
categories:
  - 漏洞研究
layout: post
published: true
---





本文对土豆家族中几个关键的提权漏洞进行了学习与分析，学习之后思路更加清晰，许多原本模糊的概念也变得明朗起来。

## Hot Potato

Hot Potato 即热土豆，是最初的土豆提权，在 2016 年由 [@breenmachine](https://x.com/breenmachine) 披露，大致流程是：

1. NBNS 欺骗
2. 构造本地 HTTP，响应 WPAD
3. HTTP -\> SMB NTLM Relay
4. 等待高权限进程的访问，即激活更新服务(低权限可激活)

### 利用过程

#### NBNS 欺骗

NBNS 是 windows 下的命名查询服务，当 DNS 解析失败时，windows 就会尝试 NBNS 解析主机名，所以当 windows 找不到 WPAD 主机时，就会利用该服务进行解析，我们此时就可以构造虚假的数据包进行欺骗，从而进行中继攻击。

> 注意：
>
> 1. 这里为了保证 dns 解析失败，使用了 UDP 端口耗尽技术，即攻击者可以伪造大量 DNS 响应包 ，并使用所有可能的源端口号进行响应，就会导致系统无法再分配新的端口用于 DNS 查询，从而我们的目标机器只能走 NBNS
> 2. 攻击者为了对 NBNS 查询进行响应，还必须匹配数据包中 TXID 字段，不过该字段只有两字节，直接暴力发送即可

#### 构造本地 HTTP，响应 WPAD

成功劫持 WPAD 之后，会向目标机器会返回一个自定义的 PAC 文件地址，（通常是 https(http)://attacker_ip/wpad.dat），该 PAC 文件中指定浏览器将所有流量代理到攻击者的机器上。文件的示例如下：

```javascript
function FindProxyForURL(url, host) {
    return "PROXY attacker_ip:80";
}
```

#### HTTP -\> SMB NTLM Relay

当用户访问某些受保护资源（如本地共享目录）时，系统会自动发起 NTLM 认证。流量经过攻击者的代理服务器，攻击者可以截获这个认证过程，然后再将认证流量转发，成功后可以获得当前用户的高权限模拟令牌,本质就是个 ntlm relay 攻击

#### 等待高权限进程的访问

Windows 更新服务（wuauserv）是以 SYSTEM 权限运行的，如果某个低权限用户能触发该服务进行网络访问（例如检查更新），它就会使用 SYSTEM 身份去访问网络资源，如果此时设置了代理为攻击者的机器，SYSTEM 用户就会向攻击者的 HTTP 服务发起 NTLM 认证请求 。

> 注意：此漏洞利用并不稳定，有时需要等待 Windows 更新和 WPAD 缓存刷新几个小时。

利用流程图如下所示：

![image](https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/image-20250602103425-ibs2cb3.png)

### 防御

目前微软已打补丁，在 MS16-075（烂土豆） 上修补跨协议中继，针对 CVE-2016-3213 的 WPAD 解析也已修复，并且在请求 PAC 文件时不再发送凭据 (CVE-2016-3236)。

## Rotten Potato

Rotten Potato 即烂土豆，同样在 2016 年由 [@breenmachine](https://x.com/breenmachine) 披露，它的优点是立即触发，而不需要进行等待。

### 前置知识

1. NTLM 是一种基于挑战-响应的身份验证协议，握手流程如下：

    |步骤|数据包类型|发送方|
    | ----| -----------------| ------|
    |1|NTLM\_NEGOTIATE|客户端|
    |2|NTLM\_CHALLENGE|服务端|
    |3|NTLM\_AUTHENTICATE|客户端|

    在 Rotten Potato 攻击中，攻击者会劫持整个握手过程，并模拟服务端完成这个流程。
2. Windows 提供了一个安全接口 API：`AcceptSecurityContext`​，用于处理 NTLM 的服务端部分。

    攻击者可以利用这个函数来：

    - 接收客户端发送的 `NTLM_NEGOTIATE`​
    - 返回 `NTLM_CHALLENGE`​
    - 接收并验证 `NTLM_AUTHENTICATE`​
    - 最终获得一个 **impersonation token（模拟令牌）**
3. 需要理解的几个知识：

    1. 使用 DCOM 时，如果以服务的方式远程连接，那么权限为 System，例如 BITS 服务
    2. 使用 DCOM 可以通过 TCP 连接到本机的一个端口，发起 NTLM 认证，该认证可以被重放
    3. LocalService 用户默认具有 SeImpersonate 和 SeAssignPrimaryToken 权限
    4. 开启 SeImpersonate 权限后，能够在调用 CreateProcessWithToken 时，传入新的 Token 创建新的进程
    5. 开启 SeAssignPrimaryToken 权限后，能够在调用 CreateProcessAsUser 时，传入新的 Token 创建新的进程

其大致原理如下所示：

1. 通过 NT AUTHORITY/SYSTEM 运行的 RPC 将尝试通过 `CoGetInstanceFromIStorage`​ API 调用向我们的本地代理进行身份验证
2. 端口 135 是 DCOM RPC 的监听端口，攻击者利用它来获取“标准”的 NTLM_CHALLENGE 数据包结构,我们使用该 `NTLM_CHALLENGE` ​回复 DCOM，但注意其中 `CHALLENGE` ​值已被替换。
3. AcceptSecurityContextAPI 调用以在本地模拟 NT AUTHORITY/SYSTEM

下面详细分析其流程。

### 利用过程

#### CoGetInstanceFromIStorage

​`CoGetInstanceFromIStorage`​，它是 COM 系统用来创建“远程对象”的标准方法，支持指定远程 IP 地址。在 RottenPotato 里，它用于“诱导 SYSTEM 服务去连接你伪造的接口”，核心就是诱导激活。

> 1. 它请求 DCOM 激活服务（SYSTEM） 帮你[初始化](https://so.csdn.net/so/search?q=%E5%88%9D%E5%A7%8B%E5%8C%96&spm=1001.2101.3001.7020) CLSID；
> 2. 系统服务以 SYSTEM 身份连接你指定的 DCOM 管道（或 RPC 接口）；

我们调用 `CoGetInstanceFromIStorage`​ 这个 api 远程激活某个 COM 对象，在利用中，通过指定恶意代理地址（可以是本地自己启动的 RPC 代理服务）来劫持这个认证过程。在原作者的 demo 中使用的 COM 对象为 BITS，CLSID 为 `{4991d34b-80a1-4291-83b6-3328366b9097}`​。

#### DCOM 向代理发送 NTLM 协商包

当 SYSTEM 身份的进程尝试连接到远程服务时，会发起 NTLM\_NEGOTIATE ，而 NTLM\_NEGOTIATE 是 NTLM 握手的第一步。

#### 代理转发

在接收到这个 `NTLM_NEGOTIATE`​ 后，攻击者的代理服务器会调用 `AcceptSecurityContext()`​ 函数，并将接收到的 `NTLM_NEGOTIATE` ​作为输入参数传递给它，`AcceptSecurityContext()`​ 会处理这个协商包，并通常会生成一个 `NTLM_CHALLENGE `​ 的 CHALLENGE 值。同时，攻击者的代理服务器会将 `NTLM_NEGOTIATE`​ 发送给本地的 135 端口，得到另一个 `NTLM_CHALLENGE`​ 数据包，将该数据包里的 CHALLENGE 值替换成 `AcceptSecurityContext()`​ 中获取的，再发送给最初的 DCOM，并且得到一个合法的 `NTLM_AUTHENTICATE`​ 数据包，然后再调用 `AcceptSecurityContext`​ 来处理这个数据包，如果成功则会返回成功状态，之后使用 `ImpersonateSecurityContext`​ 这个 api 就会获取模拟令牌。

利用流程图如下所示：

![image](https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/image-20250602133120-zii8us1.png)

再偷一个 project zero 的图（图里面的有关术语在 Rogue Potato 时进行说明）：

![image](https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/image-20250604161534-o0dlbkd.png)

### 防御

不再有效，由于 DCOM 和 OXID 解析器上的补丁，**Windows 10 1809** 和 **Windows Server 2019** 之后无法正常工作。

更加详细的分析可以参考：[Rotten Potato – 从服务账户到 SYSTEM 的权限提升 (foxglovesecurity.com)](https://foxglovesecurity.com/2016/09/26/rotten-potato-privilege-escalation-from-service-accounts-to-system/)。

## Juicy Potato

该提权是在上面的烂土豆基础上改进而来的，其基本原理是差不多的。

### 介绍

- CLSID 是标识 COM 类对象的全局唯一标识符。它是一个类似 UUID 的标识符。
- 程序员和系统管理员使用后台智能传输服务 (BITS)从 HTTP Web 服务器和 SMB 文件共享下载文件或将文件上传到 HTTP Web 服务器和 SMB 文件共享。关键是 BITS 实现了 IMarshal 接口并允许代理声明强制 NTLM 身份验证。

Rotten Potato 的 PoC 使用带有默认 CLSID 的 BITS

```c#
// Use a known local system service COM server, in this cast BITSv1
Guid clsid = new Guid("4991d34b-80a1-4291-83b6-3328366b9097");
```

但是现在发现除了 BITS 之外，还有几个进程外 COM 服务器由可能被滥用的特定 CLSID 标识。他们至少需要：

- 可由当前用户实例化，通常是具有模拟权限的服务用户
- 实现 IMarshal 接口
- 以提升的用户身份运行（SYSTEM、Administrator，...）

具体可以参考这里：[https://ohpe.it/juicy-potato/CLSID/](https://ohpe.it/juicy-potato/CLSID/)。

此外，Juicy Potato 还对创建进程方式进行了优化，如果 LocalService 开启 SeImpersonate 权限，则调用 CreateProcessWithToken 创建 system 权限进程；如果 LocalService 开启 SeAssignPrimaryToken 权限，调用 CreateProcessAsUser 创建 system 权限进程。

除此之外和利用方式和烂土豆是一样的，不再赘述。

### 防御

不再有效，同 Rotten Potato。微软修复了与 OXID 解析器相关的提权漏洞，使得攻击者无法再通过指定自定义端口（如 1337）伪造本地 RPC 服务。现在 OXID Resolver 只能使用固定的端口 135，且尝试通过远程 OXID 解析器将请求转发到本地伪造服务时，只会以 ANONYMOUS LOGON 身份运行，无法获得高权限令牌。

## Rogue Potato

RoguePotato 是在早期的 RottenPotato 和 JuicyPotato 方法基础上发展起来的，特别针对 JuicyPotato 无法工作的环境，例如 WindowsServer2019 和 WindowsBuild1809 之后的版本，微软对 DCOM 解析器进行了安全更新，强制OXID解析的端口是135并且还是匿名登录，限制了 DCOM 服务与本地 RPC 进行通信，这是为了阻止像 RottenPotato 或 JuicyPotato 这样的攻击，为了绕过这个限制，RoguePotato 使用其他远程主机的 135 端口做转发，通过远程主机将数据传到本地伪造的 RPC 服务上。

### 前置知识

1. RPCSS：RPCSS 服务是 COM 和 DCOM 服务器的服务控制管理器。它执行 COM 和 DCOM 服务器的对象激活请求、对象导出程序解析和分布式垃圾回收。

    ![image](https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/image-20250604112018-yyr7r4m.png)
2. OXID（对象导出器标识符）是一个用于标识网络上 DCOM 对象的唯一数字。

    当一个客户端应用程序想要访问一个远程 COM 对象时，它需要使用 OXID 查询来获取对象所在服务器的信息（绑定信息），随即 RPC 服务会调用 `ResolveOxid2` ​函数来解析 OXID 的查询请求，并返回绑定信息
3. 从 OBJREF 结构中连接到原始对象是一个分为两个步骤的过程：

    1. 客户端从结构中提取 对象导出器 ID（OXID） ，并根据 OBJREF 中指定的 RPC 绑定信息 ，联系对应的 OXID 解析服务（OXID Resolver） 。
    2. 客户端使用 OXID 解析服务来查找承载该对象的 COM 服务器 的 RPC 绑定信息 ，然后与该 COM 服务器的 RPC 端点建立连接，以访问对象的接口。

### 利用过程

在先前的 Rotten/Juicy Potato 攻击中，通过 CoGetInstanceFromIStorage，通过构造引用对象使指定 CLSID 的 COM 组件连接恶意服务端进行认证，这里访问的服务端就是 OXID resolver，其会解析需要加载的 Instance bind 在什么地方，在这个访问环节中存在 NTLM 认证，Rotten potato 就是在和 OXID resolver 交互的过程中进行 NTLM 中继，但是由于微软的修复，不能指定 135 以外的端口 + 访问远程使用匿名登录，烂土豆失效。rogue potato 的想法就是不在查询环节进行利用，而是通过更改查询的结果，通过实现 ResolveOxid2 方法对解析结果进行控制，将 COM 组件再次重定向回本地的恶意服务。

#### 触发 COM 服务连接远程机器 135 端口

同样的，RoguePotato 首先选择一个 CLSID(类标识符)来发起一个 DCOM 对象的激活请求，这个请求的目的是让系统创建或激活一个指定的 COM 对象。在 `IStorage` ​对象中，指定了远程 OXID 解析器的字符串绑定，这个绑定将指向我们远程的恶意 oxid 解析器的 IP 地址。当使用 `CoGetInstanceFromIStorage` ​函数对 `IStorage` ​对象进行 UnMarshall（解封）时, 会触发 DCOM 激活服务（RPCSS 中的一部分）向 oxid 解析器发送一个 oxid 解析请求, 以此定位对象的绑定信息，由于微软限制了不能指定 135 以外的端口，我们这里指定了远程机器的 135 端口。

#### 伪造 ResolveOxid2

在远程机器的 135 端口上设置一个端口转发，将所有流量转发到部署的恶意 OXID 解析服务上。然后编写恶意的 ResolveOxid2 函数的代码以此返回一个被篡改后的响应，此响应包含的绑定信息为: `ncacn_np:localhost/pipe/roguepotato[\pipe\epmapper]`​

在此绑定信息中, RoguePotato 特意使系统使用 `RPC over SMB`​(ncacn\_np), 而不是默认的 `RPC over TCP`​(ncacn\_ip\_tcp)，这是因为 SMB 协议允许通过命名管道进行通信，而命名管道可以用于接下来的权限模拟操作，而原作者实践过程中发现 ncacn_ip_tcp 返回的是识别令牌，无法利用。

#### 身份模拟

RoguePotato 在目标系统上创建了一个特殊的命名管道，其完整名称为 `\\.\pipe\roguepotato\pipe\epmapper`​，以此来等待 RPCSS 的连接，当 RPCSS 连接后，则调用 `ImpersionateNamedPipeClient` ​函数模拟 RPCSS 服务的安全上下文，这样就能以相同的权限执行代码。

#### 令牌窃取与进程创建

当攻击者的线程成功使用 `RpcImpersonateClient`​ 模拟 rpcss 服务身份后，通过枚举系统的所有进程句柄来找到 rpcss 服务的句柄，然后筛选出进程中拥有 SYSTEM 权限的令牌，最终使用 `CreateProcessAsUser`​ 或 `CreateProcessWithToken`​ 函数来创建高权限的进程

#### 关于 NETWORK_SERVICE to SYSTEM

作者在原文中写到：

> if you can trick the “Network Service” account to write to a named pipe over the “network” and are able to impersonate the pipe, you can access the tokens stored in RPCSS service
>
> 翻译：如果您可以欺骗 “Network Service” 帐户通过 “network” 写入命名管道并能够模拟该管道，则可以访问 RPCSS 服务中存储的令牌

关于为什么让 rpcss 连接到自己的恶意命名管道就能提权，可以参考文章：[从 NETWORK SERVICE 到 SYSTEM – Decoder 的博客](https://decoder.cloud/2020/05/04/from-network-service-to-system/) 以及 [Tyranid&apos;s Lair：共享登录会话有点太多 (tiraniddo.dev)](https://www.tiraniddo.dev/2020/04/sharing-logon-session-little-too-much.html)，这里简单说一下：

- **登录会话共享**：Windows 的登录会话机制允许同一会话中的不同进程共享某些权限和令牌（Token）。当 LSASS（Local Security Authority Subsystem Service）为一个登录会话创建第一个令牌时，该令牌会被存储并用于后续的网络认证。这意味着，NETWORK SERVICE 账户的进程可能共享同一个登录会话的令牌。
- **本地回环认证**：当使用 SMB（Server Message Block）协议通过本地回环（如 \\localhost\pipe\...）访问命名管道时，系统会在内核模式下执行网络认证。由于内核模式具有 TCB（Trusted Computing Base）特权，认证过程会使用登录会话中存储的第一个令牌，而这个令牌可能属于高权限进程（如 RPCSS 的 SYSTEM 令牌）。

- RPCSS（Remote Procedure Call Subsystem）是 Windows 的核心服务，运行在 SYSTEM 权限下，且通常是 NETWORK SERVICE 登录会话中的第一个进程。因此，LSASS 存储的该会话的令牌是 RPCSS 的 SYSTEM 令牌。所以在本地回环认证中，系统会使用登录会话的第一个令牌（即 SYSTEM 令牌）来完成认证，而不是调用者的实际令牌（NETWORK SERVICE 令牌）。这是漏洞的核心所在。
- 通过构造恶意命名管道，攻击者可以诱导 RPCSS 连接到该管道，触发本地回环认证，进而获取 SYSTEM 权限的令牌。

利用流程图：

![image](https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/image-20250605112643-ybtqwaf.png)

再偷一个 project zero 的图：

![image](https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/image-20250605112718-iekadse.png)

### 防御

现在仍然有效。

参考文章：[没有 JuicyPotato？老故事，欢迎 RoguePotato！– 解码器的博客 (decoder.cloud)](https://decoder.cloud/2020/05/11/no-more-juicypotato-old-story-welcome-roguepotato/)

## PrintSpoofer (or PipePotato or BadPotato)

这个漏洞有三个名字，最初公开 POC 的老外叫它 PrintSpoofer，之后 360 的 paper 叫它 PipePotato，然后 Beichen 师傅的 POC 又叫它 BadPotato，后文统一称为 PrintSpoofer。

### 利用

从前面的漏洞分析中我们可以明确，为了在另一个用户的上下文中创建进程，我们需要一个令牌，模拟令牌的等级需要为 `SecurityImpersonation`​，不同的模拟令牌区别可见下表。然后，通过一个利用命名管道模拟的服务器应用程序，我们可以获得该令牌。所以我们现在只需要找到一个 system 权限的进程去连接该命名管道就行。

|​`SecurityAnonymous`​|0|客户端身份不可见，完全匿名|
| ----| -| --------------------------|
|​**​`SecurityIdentification`​**​|**1**|**可识别客户端身份，但不能模拟其执行操作**|
|**​`SecurityImpersonation`​**​|**2**|**可以在本机上模拟客户端身份执行操作**|
|​**​`SecurityDelegation`​**​|**3**|**可以在远程机器上模拟客户端身份（需要 Kerberos 和约束委派）**|

​`spoolsv.exe` ​服务有一个公开的 RPC 服务，里面有这个函数 `RpcRemoteFindFirstPrinterChangeNotificationEx`​，此函数会创建一个远程更改通知对象，用于监视打印机对象的更改，并使用 `RpcRouterReplyPrinter ` ​或 `RpcRouterReplyPrinterEx ` ​将更改通知发送到打印客户端，该函数声明如下，其中 `pszLocalMachine` ​参数需要传递 UNC 路径，传递 `\\127.0.0.1` ​时，服务器会访问 `\\127.0.0.1\pipe\spoolss`​。

```c
DWORD RpcRemoteFindFirstPrinterChangeNotificationEx( 
    /* [in] */ PRINTER_HANDLE hPrinter,
    /* [in] */ DWORD fdwFlags,
    /* [in] */ DWORD fdwOptions,
    /* [unique][string][in] */ wchar_t *pszLocalMachine,
    /* [in] */ DWORD dwPrinterLocal,
    /* [unique][in] */ RPC_V2_NOTIFY_OPTIONS *pOptions)

```

但是 `\\127.0.0.1\pipe\spoolss` ​这个命名管道已经被 `NT AUTHORITY\SYSTEM` ​创建，并且我们希望只在本地进行利用，所以这里就利用了一个小 trick，我们 `\\127.0.0.1/pipe/foo` ​时，校验路径时会认为 `127.0.0.1/pipe/foo` ​是主机名，随后在连接 named pipe 时会对参数做标准化，将 `/` ​转化为 `\`​，于是就会连接 `\\127.0.0.1\pipe\foo\pipe\spoolss`​，攻击者就可以注册这个 named pipe 从而窃取 client 的 token，从而进行模拟。

### 防御

itm4n 在 blog 中提到，在使用 `CreateFile` ​打开命名管道时，可以添加 `SECURITY_IDENTIFICATION ` ​flag 使得模仿时得到的 token 是 identification token，即 `SecurityIdentification`​，此时获取的 token 就不能用于模拟。

但现在仍没有官方补丁。

参考文章：[PrintSpoofer - Abusing Impersonation Privileges on Windows 10 and Server 2019 | itm4n&apos;s blog](https://itm4n.github.io/printspoofer-abusing-impersonate-privileges/)

## PrintNotifyPotato/JuicyPotatoNG

这两个名字也是一个东西，只不过一个是 [BeichenDream](https://github.com/BeichenDream/PrintNotifyPotato) 师傅的 C#实现，一个是 [antonioCoco](https://github.com/antonioCoco/JuicyPotatoNG) 的 C++ 实现。

在 JuicyPotato 发布后，Microsoft 通过将获取的令牌更改为 Indentification 令牌，对可滥用的 CLSID 进行了重要修改。此外，需要属于 INTERACTIVE 组才能利用其他 CLSID（例如 PrintNotify），这并不常见。

在 [The impersonation game](https://decoder.cloud/2020/05/30/the-impersonation-game/) 中解释了为什么 RPC 调用中得到的 token 是 `identification token`​，确实是发起 RPC 调用时设定好的，并通过寻找注册表，发现了一个名为 `PrintNotify` ​的服务，其注册表中的 `Impersonation level` ​为 `impersonation`​，使用该 CLSID 对 fake OXID resolver 进行查询，重定向到本地 evil RPC server 后，在第一次 RPC 远程调用得到 anonymous logon 后，由查询 `IremUnknown2` ​触发的回调成功拿到了 SYSTEM，然而这个组件需要用户在 `INTERACTIVE` ​组中才能完全利用。
后续在 [Giving JuicyPotato a second chance: JuicyPotatoNG](https://decoder.cloud/2022/09/21/giving-juicypotato-a-second-chance-juicypotatong/) 中给出了更完整的利用，使用 `LogonUser` ​函数进行登录，由于使用的是 `LogonNewCredentials`​，LSASS 会直接给这个 token 加一个 `INTERACTIVE`​，由于这个 token 只适用于远程网络认证，所以随便填一个用户名密码均可成功。

### 本地利用

上面说的利用方式无法再本地进行利用，针对本地利用，James Forshaw 专门写了一篇文章进行说明：[Project Zero: Windows Exploitation Tricks: Relaying DCOM Authentication](https://googleprojectzero.blogspot.com/2021/10/windows-exploitation-tricks-relaying.html)。

本次攻击的最终目标是实现一种不依赖特权用户登录的本地权限提升。其核心思路是，通过一系列技术手段，在本地捕获计算机自身域账户的认证凭据，并将其成功中继到域控制器（DC）的LDAP服务，进而获取域内的高级权限。接下来对攻击流程以及几个关键点进行说明。

#### **强制本地COM使用TCP协议通信**

我们在之前的 RoguePotato 利用中，指定 OXID 查询为远程 135 端口，然后进行转发到本地恶意 OXID 解析服务进行利用，现在我们要完全在本地进行利用，所以不能走这一套了。所以我们的 OBJREF 变成了 Objref Moniker，它是一个特殊的字符串格式，像这样 objref:TUVP...，它不再是“间接”的 OBJREF，而是“直接”的 COM 对象引用，它会告诉系统我不需要走 OXID Resolver 那一套流程，我已经知道你要连接谁了。

#### **过RPCSS服务的防火墙安全检查**

- **问题**：当客户端尝试通过TCP连接COM服务端时，RPCSS服务会作为前置检查者。它会调用内部函数`IsPortOpen`​，获取发起请求的COM服务器进程的完整可执行文件路径（`ImageFileName`​），并检查该路径是否在Windows防火墙策略中被允许监听端口。对于一个未知的攻击程序，此检查必然失败，RPCSS会拒绝返回TCP绑定信息，导致客户端连接错误。
- **解决方案**：伪造进程环境块（PEB）中的进程路径。

  1. 通过测试发现，`C:\Windows\System32\svchost.exe`​ 是一个默认被防火墙策略信任的路径。
  2. 攻击者在自己的程序代码中，于初始化COM组件之前，调用API修改当前进程PEB内的`ImagePathName`​字段，将其值更改为`C:\Windows\System32\svchost.exe`​。
  3. 随后程序初始化COM并向RPCSS注册。RPCSS在注册时获取到的就是这个伪造的、受信任的路径。
  4. 当`IsPortOpen`​检查发生时，它检查的是这个伪造路径，因此检查通过，RPCSS正常返回TCP绑定信息。

#### **捕获客户端认证凭据**

- **问题**：TCP连接通路已建立，需要在服务端捕获客户端发送过来的认证数据。直接挂钩（Hook）相关API函数比较困难，且风险较高。
- **解决方案**：**修改内存中可写的安全函数表**。

  1. RPC运行时通过调用`InitSecurityInterface`​函数来从安全库（如`sspicli.dll`​）中获取一个函数表，该表包含了一系列用于处理认证的函数指针。
  2. 经分析，该函数表所在的内存区域是**可写的**。
  3. 攻击者在程序中提前调用`InitSecurityInterface`​获取到该表的地址，然后直接修改表中的函数指针，使其指向攻击者自己实现的代码。当认证发生时，系统就会调用被替换后的函数，从而截获凭据。

#### **本地中继的失败与攻击目标的转移**

- **问题**：攻击者最初尝试将截获的凭据中继回本机上的其他高权限服务，以实现本地提权。但该尝试失败了。
- **失败原因**：微软引入了专门的防御措施。RPC运行时的`SSECURITY_CONTEXT::ValidateUpgradeCriteria`​函数会检测认证请求是否同时满足两个条件：1. 来自**本机环回（Loopback）地址；2. 认证级别低于数据包完整性（**​**​`RPC_C_AUTHN_LEVEL_PKT_INTEGRITY`​**​ **）** 。如果都满足，该连接将被视为不安全并被拒绝。
- **策略变更**：既然本地中继的路被阻塞，攻击策略转向网络中继。
- **新目标**：选择**域控制器的LDAP服务**。选择它的原因是，LDAP服务的默认配置**不强制要求客户端进行LDAP签名**。这对于中继攻击至关重要，因为攻击者只有认证凭据，没有建立通信所需的会话密钥，因此无法生成签名。

#### **修改系统LDAP库的行为**

- **问题**：为了与LDAP服务通信，攻击者决定使用系统自带的`wldap32.dll`​库。但该库在默认情况下会尝试启用LDAP签名，这会导致中继攻击失败。控制这一行为的注册表键`LdapClientIntegrity`​需要管理员权限才能修改。
- **解决方案**：**使用**​**​`RegOverridePredefKey`​**​ **API临时重定向注册表查询**。

  1. 攻击者调用`RegOverridePredefKey`​，将所有对`HKEY_LOCAL_MACHINE`​注册表根键的查询，临时重定向到攻击者在`HKEY_CURRENT_USER`​下创建的一个可控键上。
  2. 在那个可控键下，攻击者仿造路径并设置`LdapClientIntegrity`​的值为`0`​（禁用签名）。
  3. 加载`wldap32.dll`​。该DLL在初始化时读取配置，由于重定向，它会读到攻击者设置的禁用签名的值。
  4. DLL加载完成后，立即取消注册表重定向，恢复系统正常状态，整个过程对其他程序无影响。

#### **选择用于触发认证的COM服务**

- **问题**：需要找到一个合适的系统服务来发起最初的认证，这个服务必须满足多个苛刻的条件。
- **筛选条件**：

  1. 必须以`SYSTEM`​或`NETWORK SERVICE`​身份运行，这样其网络认证才会使用计算机账户。
  2. 其COM认证级别必须是`RPC_C_AUTHN_LEVEL_PKT_CONNECT`​，以确保不启用完整性保护。
  3. 不能是托管在`svchost.exe`​中的服务，因为`svchost.exe`​会统一提升其托管服务的安全级别。
- **选定的触发器**：`CRemoteAppLifetimeManager`​ COM服务。该服务完全符合上述所有条件。

完整的攻击流程：

- 攻击者在目标机器上运行其综合性攻击程序。
- 程序设置好安全函数表劫持和LDAP库改造的陷阱。
- 程序创建一个恶意的COM对象引用（OBJREF），并诱使第4步中找到的`CRemoteAppLifetimeManager`​服务来解析它。
- ​`CRemoteAppLifetimeManager`​服务（以`NETWORK SERVICE`​身份）在解析时，向攻击者的陷阱发起了认证。由于是网络认证，它使用的是计算机账户的凭据。
- 陷阱通过被劫持的安全函数表，成功捕获到计算机账户的Kerberos凭据。
- 攻击程序立即将此凭据中继给被改造过的`wldap32.dll`​库。
- ​`wldap32.dll`​带着这个高权限凭据，向域控制器的LDAP服务发起连接（由于改造，连接不要求签名）。
- 连接成功，攻击者现在能以计算机账户的身份对域控LDAP执行任意操作，例如将自己的账户加入域管理员组，从而完全控制整个域。

### 防御

目前仍然有效

## GodPotato

Beichen师傅的作品，一个通过 DCOM 提权的方式。利用了 Windows rpcss 服务对 OXID 处理的漏洞进行提权操作。

### 利用

#### **初始化与RPC接口定位**

攻击的准备阶段，目标是为后续的函数挂钩定位必要的信息。

1. **定位RPC接口结构**: 程序首先会加载并分析核心COM组件库 `combase.dll`​。它会在该模块内搜索一个特定的接口GUID (`18f70770-8e64-11cf-9af1-0020af6e72f4`​)，以准确定位到DCOM通信所依赖的 `RPC_SERVER_INTERFACE`​ 结构体。这个结构体包含了RPC调度的关键信息。
2. **解析目标函数**: 找到接口后，程序会进一步解析其定义，锁定一个名为 `_UseProtSeq`​ 的函数。此函数在DCOM中负责选择通信协议序列，是后续进行调用劫持的理想目标。

#### **关键函数挂钩 (** _UseProtSeq **)**

这是实现调用劫持的核心步骤，通过在内存中修改函数指针，改变程序的执行流程。

1. **创建代理函数**: 利用C#的委托（Delegate）机制，创建一个与原始 `_UseProtSeq`​ 函数具有完全相同签名（参数列表、返回值类型）的代理函数。所有恶意的重定向逻辑都将在这个代理函数中实现。
2. **修改RPC调度表**: 程序调用 `VirtualProtect`​ API来获取目标内存区域（RPC调度表）的写权限。随后，通过 `Marshal.WriteIntPtr`​ 等方法，将调度表中记录的原始 `_UseProtSeq`​ 函数地址，强行覆盖为我们创建的代理函数的入口地址。
3. **完成劫持**: 经过此步骤，任何高权限服务（如 `RPCSS`​）在后续的DCOM通信中对 `_UseProtSeq`​ 的正常调用，都将被重定向到我们的代理函数，从而落入攻击者的控制之下。

#### **RPC重定向与身份模拟**

这是漏洞利用的触发和核心环节，将高权限的RPC调用引导至攻击者掌控的通道中。

1. **创建命名管道 (Named Pipe)** : 攻击者在本地 一个自定义的命名管道，例如 `\\.\pipe\GodPotato\pipe\epmapper`​。这个管道将作为接收重定向RPC调用的服务端。
2. **触发并重定向RPC调用**: 当被挂钩的代理函数（`_UseProtSeq`​）被高权限服务调用时，它会中断原始的执行流，转而将该服务的RPC通信请求，强制重定向到上一步创建的命名管道。
3. **调用** `ImpersonateNamedPipeClient`​: 作为命名管道的服务器，攻击者程序会接收到来自高权限服务的连接。在连接建立的瞬间，程序会立刻调用Windows API `ImpersonateNamedPipeClient`​。这是整个攻击链的精髓，该函数使得当前线程能够完全模拟（Impersonate）管道客户端（即高权限服务）的安全上下文。

#### **令牌窃取与进程创建**

成功模拟身份后，最后一步就是利用这个临时的、高权限的身份来完成最终的提权。

1. **获取高权限令牌 (Token)** : 在`Impersonate`​成功后，当前线程便在操作系统的调度层面暂时拥有了 `SYSTEM`​ 权限。程序会立即从当前线程中提取其安全令牌（Access Token），这个令牌中包含了 `SYSTEM`​ 账户的所有权限信息。
2. **创建新进程**: 最后，攻击者利用这个窃取到的 `SYSTEM`​ 令牌，调用 `CreateProcessAsUser`​ 或 `CreateProcessWithTokenW`​ 等API，以系统的最高权限创建一个新的进程，例如 `cmd.exe`​。

详细的源码分析可以参考：[【RPC】GodPotato 原理分析 - oone (holdyounger.github.io)](https://holdyounger.github.io/2025/01/09/A_OS/Windows/RPC/%E3%80%90RPC%E3%80%91GodPotato%E5%8E%9F%E7%90%86%E5%88%86%E6%9E%90/)

## 其他参考文章

[Potato 家族提权分析 - Geekby&apos;s Blog](https://www.geekby.site/2020/08/potato%E5%AE%B6%E6%97%8F%E6%8F%90%E6%9D%83%E5%88%86%E6%9E%90/)

[In the Potato family|HideAndSec](https://hideandsec.sh/books/windows-sNL/page/in-the-potato-family-i-want-them-all)

[Potato 家族本地提权细节-先知社区 (aliyun.com)](https://xz.aliyun.com/news/7371)

[奇安信攻防社区-Potato 提权合集 (butian.net)](https://forum.butian.net/share/860)

[RedTeamNotes/土豆提权原理.md at master · xf555er/RedTeamNotes (github.com)](https://github.com/xf555er/RedTeamNotes/blob/master/%E5%9C%9F%E8%B1%86%E6%8F%90%E6%9D%83%E5%8E%9F%E7%90%86.md)

[从烂土豆开始的土豆家族入门 | Z3ratu1&apos;s blog](https://blog.z3ratu1.top/%E4%BB%8E%E7%83%82%E5%9C%9F%E8%B1%86%E5%BC%80%E5%A7%8B%E7%9A%84%E5%9C%9F%E8%B1%86%E5%AE%B6%E6%97%8F%E5%85%A5%E9%97%A8.html)

[Potatoes - Windows Privilege Escalation · Jorge Lajara Website (jlajara.gitlab.io)](https://jlajara.gitlab.io/Potatoes_Windows_Privesc)

[Project Zero：Windows 漏洞利用技巧：中继 DCOM 身份验证 (googleprojectzero.blogspot.com)](https://googleprojectzero.blogspot.com/2021/10/windows-exploitation-tricks-relaying.html)
