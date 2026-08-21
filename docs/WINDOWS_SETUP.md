# Подготовка Windows для Docker + RTX 3080 Ti

На текущем компьютере диагностика показала:

- `VirtualizationFirmwareEnabled: False`;
- `HypervisorPresent: False`;
- Docker: `HCS_E_HYPERV_NOT_INSTALLED`.

Сначала войдите в BIOS/UEFI и включите аппаратную виртуализацию:

- для AMD настройка обычно называется `SVM Mode`;
- для Intel — `Intel Virtualization Technology`, `VT-x` или `VMX`.

Сохраните настройки и загрузите Windows. Затем откройте PowerShell **от имени администратора** и выполните:

```powershell
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
bcdedit /set hypervisorlaunchtype auto
wsl --update
```

Перезагрузите компьютер. После перезагрузки:

```powershell
wsl --status
wsl --shutdown
```

Запустите Docker Desktop, выберите Linux containers и включите WSL Integration для Ubuntu-22.04. Проверка:

```powershell
docker version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Обе команды должны завершиться без `HCS_E_HYPERV_NOT_INSTALLED`; во второй должна отображаться RTX 3080 Ti.
