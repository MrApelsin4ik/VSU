from email.policy import default

from django.contrib.auth.models import AbstractBaseUser, BaseUserManager, PermissionsMixin
from django.db import models


class CustomUserManager(BaseUserManager):
    def create_user(self, email, username='', password=None):
        if not email:
            raise ValueError("Email обязателен")

        email = self.normalize_email(email)
        user = self.model(email=email, username=username)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, username='', password=None):
        user = self.create_user(email, username, password)
        user.is_staff = True
        user.is_superuser = True
        user.save(using=self._db)
        return user


class CustomUser(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True)
    username = models.CharField(max_length=50, unique=False, blank=True, default='')
    full_name = models.CharField(max_length=70, blank=True, default='')
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    objects = CustomUserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = []  # чтобы Django не требовал доп. полей при создании суперюзера

    def __str__(self):
        return self.username or self.email


class Section(models.Model):
    """
    Разделы и подразделы (рекурсивная иерархия).
    """

    class SectionType(models.TextChoices):
        MAIN = "main", "Главное"
        NEWS = "news", "Новости"
        ANNOUNCEMENT = "announcement", "Объявления"
        COURSES = "courses", "Курсы"

    title = models.CharField("Название раздела", max_length=255)
    parent = models.ForeignKey(
        "self",
        verbose_name="Родительский раздел",
        on_delete=models.CASCADE,
        null=True,
        blank=True,
        related_name="children",
    )
    section_type = models.CharField(
        "Принадлежность",
        max_length=20,
        choices=SectionType.choices,
        default=SectionType.MAIN,
    )
    relative_url = models.CharField("Относительная ссылка", max_length=255, blank=True, null=True)

    class Meta:
        verbose_name = "Раздел"
        verbose_name_plural = "Разделы"

    def __str__(self):
        return self.title

class Course(models.Model):
    """
    Курсы: привязка к разделу, название, описание.
    """
    section = models.ForeignKey(
        Section,
        verbose_name="Раздел",
        on_delete=models.PROTECT,
        related_name="courses",
        null=True,
        blank=True,
    )
    title = models.CharField("Название курса", max_length=255)
    description = models.TextField("Описание", blank=True)
    description_ai = models.TextField(blank=True, default='')
    description_ai_l2 = models.TextField(blank=True, default='')
    created_at = models.DateTimeField("Дата создания", auto_now_add=True)
    updated_at = models.DateTimeField("Дата изменения", auto_now=True)
    is_active = models.BooleanField("Активен", default=True)

    class Meta:
        verbose_name = "Курс"
        verbose_name_plural = "Курсы"
        ordering = ("title",)

    def __str__(self):
        return self.title

    def preview_image(self):
        """
        Возвращает объект CourseImage, помеченный как превью (если есть),
        иначе первый по sort_order, иначе None.
        Удобно для показа в админке/листинге.
        """
        preview = self.images.filter(is_preview=True).first()
        if preview:
            return preview
        return self.images.order_by("sort_order").first()


class CourseImage(models.Model):
    """
    Изображения, привязанные к курсу (для краткого описания/превью и т.п.).
    При установке is_preview=True, остальные изображения этого курса будут сброшены.
    """
    course = models.ForeignKey(
        Course,
        verbose_name="Курс",
        on_delete=models.CASCADE,
        related_name="images",
    )
    image = models.ImageField("Изображение", upload_to="courses/images/")
    is_preview = models.BooleanField("Использовать как превью", default=False)
    sort_order = models.PositiveIntegerField("Порядок", default=0)

    class Meta:
        verbose_name = "Изображение курса"
        verbose_name_plural = "Изображения курсов"
        ordering = ("sort_order", "id")

    def __str__(self):
        return f"Image for {self.course.title}"

    def save(self, *args, **kwargs):
        # если помечаем это изображение как превью, снимаем флаг с других изображений курса
        super_save_needed = True
        if self.is_preview:
            # обновляем остальные записи в БД, чтобы у курса было только одно превью
            CourseImage.objects.filter(course=self.course, is_preview=True).exclude(pk=self.pk).update(is_preview=False)
        super().save(*args, **kwargs)


class CourseAttachment(models.Model):
    """
    Файлы, прикреплённые к курсу.
    """
    course = models.ForeignKey(
        Course,
        verbose_name="Курс",
        on_delete=models.CASCADE,
        related_name="attachments",
    )
    file = models.FileField("Файл", upload_to="courses/attachments/")
    original_name = models.CharField("Оригинальное имя файла", max_length=255, blank=True)

    class Meta:
        verbose_name = "Файл курса"
        verbose_name_plural = "Файлы курсов"

    def __str__(self):
        return self.original_name or self.file.name

    def save(self, *args, **kwargs):

        if not self.original_name:
            self.original_name = self.file.name.split('/')[-1]
        super().save(*args, **kwargs)


class News(models.Model):
    """
    Новость: тема, краткое описание + изображения, подробное описание (текст + base64-изображения),
    прикреплённые файлы, привязка к разделу/подразделу.
    """

    section = models.ForeignKey(
        Section,
        verbose_name="Раздел",
        on_delete=models.PROTECT,
        related_name="news",
    )
    title = models.CharField("Тема", max_length=255)
    short_description = models.TextField("Краткое описание")
    body = models.TextField(blank=True, default='')
    description_ai = models.TextField(blank=True, default='')
    description_ai_l2 = models.TextField(blank=True, default='')
    created_at = models.DateTimeField("Дата создания", auto_now_add=True)
    updated_at = models.DateTimeField("Дата изменения", auto_now=True)
    is_published = models.BooleanField("Опубликовано", default=True)

    class Meta:
        verbose_name = "Новость"
        verbose_name_plural = "Новости"
        ordering = ("-created_at",)

    def __str__(self):
        return self.title



class NewsImage(models.Model):
    """
    Изображения, привязанные к новости (для краткого описания/превью и т.п.).
    """

    news = models.ForeignKey(
        News,
        verbose_name="Новость",
        on_delete=models.CASCADE,
        related_name="images",
    )
    image = models.ImageField(
        "Изображение",
        upload_to="news/images/",
    )
    is_preview = models.BooleanField(
        "Использовать как превью",
        default=False,
    )
    sort_order = models.PositiveIntegerField(
        "Порядок",
        default=0,
    )

    class Meta:
        verbose_name = "Изображение новости"
        verbose_name_plural = "Изображения новостей"
        ordering = ("sort_order", "id")

    def __str__(self):
        return f"Изображение для: {self.news.title}"


class NewsAttachment(models.Model):
    """
    Файлы, прикреплённые к новости
    """

    news = models.ForeignKey(
        News,
        verbose_name="Новость",
        on_delete=models.CASCADE,
        related_name="attachments",
    )
    file = models.FileField(
        "Файл",
        upload_to="news/attachments/",
    )
    original_name = models.CharField(
        "Оригинальное имя файла",
        max_length=255,
        blank=True,
    )

    class Meta:
        verbose_name = "Файл новости"
        verbose_name_plural = "Файлы новостей"

    def __str__(self):
        return self.original_name or self.file.name

    def save(self, *args, **kwargs):
        if not self.original_name:
            self.original_name = self.file.name.split('/')[-1]
        super().save(*args, **kwargs)


class Announcement(models.Model):
    """
    Объявление: тема, описание + изображения + файлы, ссылка для перехода.
    """

    section = models.ForeignKey(
        Section,
        verbose_name="Раздел",
        on_delete=models.PROTECT,
        related_name="announcements",
        null=True,
        blank=True,
    )
    title = models.CharField("Тема", max_length=255)
    body = models.TextField("Описание", blank=True, default='')
    description_ai = models.TextField(blank=True, default='')
    description_ai_l2 = models.TextField(blank=True, default='')
    created_at = models.DateTimeField("Дата создания", auto_now_add=True)
    updated_at = models.DateTimeField("Дата изменения", auto_now=True)
    is_active = models.BooleanField("Активно", default=True)

    class Meta:
        verbose_name = "Объявление"
        verbose_name_plural = "Объявления"
        ordering = ("-created_at",)

    def __str__(self):
        return self.title


class AnnouncementImage(models.Model):
    """
    Изображения, привязанные к объявлению.
    """

    announcement = models.ForeignKey(
        Announcement,
        verbose_name="Объявление",
        on_delete=models.CASCADE,
        related_name="images",
    )
    image = models.ImageField(
        "Изображение",
        upload_to="announcements/images/",
    )
    sort_order = models.PositiveIntegerField("Порядок", default=0)

    class Meta:
        verbose_name = "Изображение объявления"
        verbose_name_plural = "Изображения объявлений"
        ordering = ("sort_order", "id")

    def __str__(self):
        return f"Изображение для: {self.announcement.title}"


class AnnouncementAttachment(models.Model):
    """
    Файлы, прикреплённые к объявлению.
    """

    announcement = models.ForeignKey(
        Announcement,
        verbose_name="Объявление",
        on_delete=models.CASCADE,
        related_name="attachments",
    )
    file = models.FileField(
        "Файл",
        upload_to="announcements/attachments/",
    )
    original_name = models.CharField(
        "Оригинальное имя файла",
        max_length=255,
        blank=True,
    )

    class Meta:
        verbose_name = "Файл объявления"
        verbose_name_plural = "Файлы объявлений"

    def __str__(self):
        return self.original_name or self.file.name

    def save(self, *args, **kwargs):
        if not self.original_name:
            self.original_name = self.file.name.split('/')[-1]
        super().save(*args, **kwargs)
